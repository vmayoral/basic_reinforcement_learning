'''
 DOOM AI player using random actions (WIP)
    @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import gym
import ppaquette_gym_doom
import random
import pandas
import os
from random import sample as rsample
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD, RMSprop
from matplotlib import pyplot as plt
import json

# GRID_SIZE = 10

# def episode():
#     """ 
#     Coroutine function for an episode.     
#         Action has to be explicitly sent (via "send") to this co-routine.
#     """
#     x, y, x_basket = (
#         np.random.randint(0, GRID_SIZE),        # X of fruit
#         0,                                      # Y of dot
#         np.random.randint(1, GRID_SIZE - 1))    # X of basket
        
#     while True:
#         # Reset grid
#         X = np.zeros((GRID_SIZE, GRID_SIZE))  
#         # Draw the fruit in the screen
#         X[y, x] = 1.
#         # Draw the basket
#         bar = range(x_basket - 1, x_basket + 2)
#         X[-1, bar] = 1.
        
#         # End of game is known when fruit is at penultimate line of grid.
#         # End represents either the reward (a win or a loss)
#         end = int(y >= GRID_SIZE - 2)
#         if end and x not in bar:
#             end *= -1

#         action = yield X[np.newaxis], end
#         if end:
#             break

#         x_basket = min(max(x_basket + action, 1), GRID_SIZE - 2)
#         y += 1

            
def experience_replay(batch_size):
    """
    Coroutine function for implementing experience replay.    
        Provides a new experience by calling "send", which in turn yields 
        a random batch of previous replay experiences.
    """
    memory = []
    while True:
        # experience is a tuple containing (S, action, reward, S_prime)
        experience = yield rsample(memory, batch_size) if batch_size <= len(memory) else None
        memory.append(experience)

def save_img():
    """
    Coroutine to store images in the "images" directory
    """
    if 'images' not in os.listdir('.'):
        os.mkdir('images')
    frame = 0
    while True:
        screen = (yield)
        plt.imshow(screen[0], interpolation='none')
        plt.savefig('images/%03i.png' % frame)
        frame += 1

nb_epochs = 50
batch_size = 128
epsilon = .8
gamma = .8

# Recipe of deep reinforcement learning model
model = Sequential()
model.add(Convolution2D(16, nb_row=3, nb_col=3, input_shape=(3, 480, 640), activation='relu'))
model.add(Convolution2D(16, nb_row=3, nb_col=3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(3))
model.compile(RMSprop(), 'MSE')
model.summary()

#################
# RELOAD A MODEL
#################
# model = model_from_json(open('model.json').read())
# model.load_weights('model.h5')

exp_replay = experience_replay(batch_size)
exp_replay.next()  # Start experience-replay coroutine

for i in xrange(nb_epochs):
    ep = episode()
    S, reward = ep.next()  # Start coroutine of single entire episode
    loss = 0.
    try:
        while True:
            action = np.random.randint(-1, 2) 
            if np.random.random() > epsilon:
                # Get the index of the maximum q-value of the model.
                # Subtract one because actions are either -1, 0, or 1
                action = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0] - 1

            S_prime, reward = ep.send(action)
            experience = (S, action, reward, S_prime)
            S = S_prime
            
            batch = exp_replay.send(experience)
            if batch:
                inputs = []
                targets = []
                for s, a, r, s_prime in batch:
                    # The targets of unchosen actions are the q-values of the model,
                    # so that the corresponding errors are 0. The targets of chosen actions
                    # are either the rewards, in case a terminal state has been reached, 
                    # or future discounted q-values, in case episodes are still running.
                    t = model.predict(s[np.newaxis]).flatten()
                    t[a + 1] = r
                    if not r:
                        t[a + 1] = r + gamma * model.predict(s_prime[np.newaxis]).max(axis=-1)
                    targets.append(t)
                    inputs.append(s)
                
                loss += model.train_on_batch(np.array(inputs), np.array(targets))

    except StopIteration:
        pass
    
    # if (i + 1) % 100 == 0:
    print 'Epoch %i, loss: %.6f' % (i + 1, loss)

#################
# SAVE THE MODEL
#################
# Save trained model weights and architecture, this will be used by the visualization code
model_name = "model.h5"
print("Saving the model to "+model_name)
json_string = model.to_json()
open('model.json', 'w').write(json_string)
model.save_weights(model_name)    

#################
# TEST
#################
img_saver = save_img()
img_saver.next()

for _ in xrange(10):
    g = episode()
    S, _ = g.next()
    img_saver.send(S)
    try:
        while True:
            act = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0] - 1
            S, _ = g.send(act)
            img_saver.send(S)

    except StopIteration:
        pass

img_saver.close()


##### New code
#NEEDS WORK


if __name__ == '__main__':
    env = gym.make('ppaquette/DoomBasic-v0')
    env.monitor.start('/tmp/doom-experiment-1', force=True)
        # video_callable=lambda count: count % 10 == 0)

    goal_average_steps = 195
    max_number_of_steps = 200
    last_time_steps = numpy.ndarray(0)
    n_bins = 8
    n_bins_angle = 10

    number_of_features = env.observation_space.shape[0]
    last_time_steps = numpy.ndarray(0)

    action_attack = [False]*43
    action_attack[0] = True

    action_right = [False]*43
    action_right[10] = True
    
    action_left = [False]*43
    action_left[11] = True

    actions = [action_attack, action_left, action_right]

    for i_episode in xrange(30):
        observation = env.reset()
        for t in xrange(max_number_of_steps):	    	
            env.render()
            # Execute the action and get feedback
            observation, reward, done, info = env.step(env.action_space.sample())
            if done:
                break


    # l = last_time_steps.tolist()
    # l.sort()
    # print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.monitor.close()
    # gym.upload('/tmp/cartpole-experiment-1', algorithm_id='vmayoral simple Q-learning', api_key='your-key')
