from collections import namedtuple
import itertools as it
import os
from random import sample as rsample

import numpy as np

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD, RMSprop

from matplotlib import pyplot as plt


GRID_SIZE = 10

Fruit = namedtuple('Fruit', ['x', 'y'])


def game(snake_length=3):
    """
    Coroutine of single snake game.

    An action tuple (dx, dy) has to be provided each call to send.
    """
    actions = [(-1, 0)] * snake_length  # An action for each snake segment
    head_x = GRID_SIZE // 2 - snake_length // 2
    snake = [(x, GRID_SIZE // 2) for x in range(head_x, head_x + snake_length)]
    grow = -1  # Don't start growing snake yet
    fruit = Fruit(-1, -1)

    while True:
        # Draw borders
        screen = np.zeros((GRID_SIZE, GRID_SIZE))
        screen[[0, -1]] = 1
        screen[:, [0, -1]] = 1
        sum_of_borders = screen.sum()

        # Draw snake
        for segm in snake:
            x, y = segm
            screen[y, x] = 1

        # Snake hit into wall or ate itself
        end_of_game = len(snake) > len(set(snake)) or \
            screen.sum() < sum_of_borders + len(snake)
        reward = -1 * end_of_game

        # Draw fruit
        if screen[fruit.y, fruit.x] > .5:
            grow += 1
            reward = len(snake)
            while True:
                fruit = Fruit(*np.random.randint(1, GRID_SIZE - 1, 2))
                if screen[fruit.y, fruit.x] < 1:
                    break

        screen[fruit.y, fruit.x] = .5

        action = yield screen, reward

        step_size = sum([abs(act) for act in action])
        if not step_size:
            action = actions[0]  # Repeat last action
        elif step_size > 1:
            raise ValueError, 'Cannot move more than 1 unit at a time'

        actions.insert(0, action)
        actions.pop()

        # For as long as the snake needs to grow,
        # copy last segment, and add (0, 0) action
        if grow > 0:
            snake.append(snake[-1])
            actions.append((0, 0))
            grow -= 1

        # Update snake segments
        for ix, act in enumerate(actions):
            x, y = snake[ix]
            delta_x, delta_y = act
            snake[ix] = x + delta_x, y + delta_y

        if end_of_game:
            break


def experience_replay(batch_size):
    """
    Coroutine of experience replay.
    
    Provide a new experience by calling send, which in turn yields 
    a random batch of previous replay experiences.
    """
    memory = []
    while True:
        experience = yield rsample(memory, batch_size) if batch_size <= len(memory) else None
        memory.append(experience)


nb_epochs = 10000
batch_size = 64
epsilon = 1.
gamma = .8

all_possible_actions = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
nb_actions = len(all_possible_actions)

nb_frames = 4  # Number of frames (i.e., screens) to keep in history

# Recipe of deep reinforcement learning model
model = Sequential()
model.add(BatchNormalization(axis=1, input_shape=(nb_frames, GRID_SIZE, GRID_SIZE)))
model.add(Convolution2D(16, nb_row=3, nb_col=3, activation='relu'))
model.add(Convolution2D(32, nb_row=3, nb_col=3, activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(nb_actions))
model.compile(RMSprop(), 'MSE')

exp_replay = experience_replay(batch_size)
exp_replay.next()  # Start experience replay coroutine

for i in xrange(nb_epochs):
    g = game()
    screen, _ = g.next()
    S = np.asarray([screen] * nb_frames)
    try:
        # Decrease epsilon over the first half of training
        if epsilon > .1:
            epsilon -= .9 / (nb_epochs / 2)

        loss = 0.
        while True:
            ix = np.random.randint(nb_actions)
            if np.random.random() > epsilon:
                ix = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0]

            action = all_possible_actions[ix]
            screen, reward = g.send(action)
            S_prime = np.zeros_like(S) 
            S_prime[1:] = S[:-1]
            S_prime[0] = screen
            experience = (S, action, reward, S_prime)
            S = S_prime
            
            batch = exp_replay.send(experience)
            if batch:
                inputs = []
                targets = []
                for s, a, r, s_prime in batch:
                    # The targets of unchosen actions are set to the q-values of the model,
                    # so that the corresponding errors are 0. The targets of chosen actions
                    # are set to either the rewards, in case a terminal state has been reached, 
                    # or future discounted q-values, in case episodes are still running.
                    t = model.predict(s[np.newaxis]).flatten()
                    ix = all_possible_actions.index(a)
                    if r < 0:
                        t[ix] = r
                    else:
                        t[ix] = r + gamma * model.predict(s_prime[np.newaxis]).max(axis=-1)
                    targets.append(t)
                    inputs.append(s)
                
                loss += model.train_on_batch(np.array(inputs), np.array(targets))
                # print(loss)

    except StopIteration:
       pass

    if (i + 1) % 100 == 0:
        print 'Epoch %6i/%i, loss: %.6f, epsilon: %.3f' % (i + 1, nb_epochs, loss, epsilon)


def save_img():
    if 'images' not in os.listdir('.'):
        os.mkdir('images')
    frame_cnt = it.count()
    while True:
        screen = (yield)
        plt.imshow(screen, interpolation='none')
        plt.savefig('images/%04i.png' % (frame_cnt.next(), ))
    

img_saver = save_img()
img_saver.next()

game_cnt = it.count(1)
for _ in xrange(10):
    g = game()
    screen, _ = g.next()
    img_saver.send(screen)
    frame_cnt = it.count()
    try:
        S = np.asarray([screen] * nb_frames)
        while True:
            frame_cnt.next()
            ix = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0]
            screen, _ = g.send(all_possible_actions[ix])
            S[1:] = S[:-1]
            S[0] = screen
            img_saver.send(screen)
            
    except StopIteration:
        print 'Saved %3i frames for game %3i' % (frame_cnt.next(), game_cnt.next())

img_saver.close()

