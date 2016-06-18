'''
DQN approach with experience replay for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning

TODO:
    - learning is just not fine
    - review bins, and activation functions

Inspired by 
        - https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA
        - http://outlace.com/Reinforcement-Learning-Part-3/

        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import gym
import numpy
import random
import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD, Adagrad, Adam

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        
        # instead of a dictionary, we'll be using
        #   a neural network
        # self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        
        # Build the neural network
        self.network = Sequential()
        self.network.add(Dense(20, init='lecun_uniform', input_shape=(4,)))
        self.network.add(Activation('sigmoid'))
        #self.network.add(Dropout(0.2))

        # self.network.add(Dense(150, init='lecun_uniform'))
        # self.network.add(Activation('sigmoid'))
        # #self.network.add(Dropout(0.2))

        self.network.add(Dense(2, init='lecun_uniform'))
        self.network.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        # rms = SGD(lr=0.001, decay=1e-6, momentum=0.5) # explodes to non
        # rms = RMSprop()
        # rms = Adagrad()
        rms = Adam()
        self.network.compile(loss='mse', optimizer=rms)
        # Get a summary of the network
        self.network.summary()

    # def learnQ(self, state, action, reward, newState, terminal=False):
    #     '''
    #     DQN learning:
    #         Instead of the Q-learning:
    #             Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
    #         we'll be updating the network following:
    #             target = reward(s,a) + gamma * max(Q(s')
    #     '''
    #     state = numpy.asarray(state)
    #     state = state.reshape(1,4)

    #     newState = numpy.asarray(newState)
    #     newState = newState.reshape(1,4)
        
    #     qval = self.network.predict(state, batch_size=1)
    #     newQ = self.network.predict(newState, batch_size=1)
    #     maxNewQ = numpy.max(newQ)
    #     y = numpy.zeros((1,2))
    #     y[:] = qval[:]
    #     if terminal:
    #         update = reward
    #     else:
    #         update = (reward + (self.gamma * maxNewQ))
    #     y[0][action] = update #target output        
    #     self.network.fit(state, y, batch_size=1, nb_epoch=1, verbose=0)

    # def chooseAction(self, state, return_q=False):        
    #     if (random.random() < self.epsilon): #choose random action
    #         action = numpy.random.randint(0,2)
    #     else: #choose best action from Q(s,a) values
    #         # convert to a numpy array
    #         state = numpy.asarray(state)
    #         state = state.reshape(1,4)
    #         # Let's run our Q function on state "state" to get Q values for all possible actions
    #         qvals = self.network.predict(state, batch_size=1)
    #         # Select the neuron that fired the most
    #         action = numpy.argmax(qvals)
    #         # print(state)
    #         # print(qvals)
    #         # print(action)
    #         q = qvals[0][action]
    #         # if return_q: # if they want it, give it!
    #         #     return action, q        
    #     return action

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # env.monitor.start('/tmp/cartpole-experiment-1', force=True)
        # video_callable=lambda count: count % 10 == 0)

    epochs = 200
    goal_average_steps = 195
    max_number_of_steps = 200
    batchSize = 100
    buffer = 500
    replay = []
    h = 0
    last_time_steps = numpy.ndarray(0)

    # Discretization of the space
    n_bins = 20
    n_bins_angle = 20

    number_of_features = env.observation_space.shape[0]
    last_time_steps = numpy.ndarray(0)

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
    pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
    cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

    # The Q-learn algorithm
    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=1)

    for i_episode in xrange(epochs):
        observation = env.reset()

        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation            
        state = [to_bin(cart_position, cart_position_bins),
                         to_bin(pole_angle, pole_angle_bins),
                         to_bin(cart_velocity, cart_velocity_bins),
                         to_bin(angle_rate_of_change, angle_rate_bins)]

        cumulated_reward = 0

        for t in xrange(max_number_of_steps):           
            env.render()

            # Pick an action based on the current state            
            # action = qlearn.chooseAction(state)

            #Let's run our Q function on S to get Q values for all possible actions
            qval = qlearn.network.predict(numpy.asarray(state).reshape(1,4), batch_size=1)
            if (random.random() < qlearn.epsilon): #choose random action
                action = numpy.random.randint(0,2)
            else: #choose best action from Q(s,a) values
                action = (numpy.argmax(qval))            

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation            
            nextState = [to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)]

            if done:
                # reward = -200
                cumulated_reward += reward
            else:
                cumulated_reward += reward


            # If out of bounds
            if (cart_position > 2.4 or cart_position < -2.4):
                # reward = -200
                print("Out of bounds, reseting")
                break

            # Experience replay storage
            if (len(replay) < buffer): #if buffer not filled, add to it
                replay.append((state, action, reward, nextState))
            else: #if buffer full, overwrite old values
                if (h < (buffer-1)):
                    h += 1
                else:
                    h = 0
                replay[h] = (state, action, reward, nextState)
                #randomly sample our experience replay memory
                minibatch = random.sample(replay, batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    #Get max_Q(S',a)
                    old_state_m, action_m, reward_m, new_state_m = memory                
                    old_qval = qlearn.network.predict(numpy.asarray(old_state_m).reshape(1,4), batch_size=1)
                    newQ = qlearn.network.predict(numpy.asarray(new_state_m).reshape(1,4), batch_size=1)
                    maxQ = numpy.max(newQ)
                    y = numpy.zeros((1,2))
                    y[:] = old_qval[:]

                    if not(done): #non-terminal state
                        update = (reward_m + (qlearn.gamma * maxQ))
                    else: #terminal state
                        # update = -200
                        update = reward_m
                    y[0][action_m] = update
                    X_train.append(numpy.asarray(old_state_m).reshape(4,))
                    y_train.append(y.reshape(2,))

                X_train = numpy.array(X_train)
                y_train = numpy.array(y_train)                
                qlearn.network.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=1)
                state = nextState
                
            if done: #if reached terminal state, update game status
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                print reward
                # print(qlearn.epsilon)
                # break

        if qlearn.epsilon > 0.1:
            qlearn.epsilon = qlearn.epsilon - (1.0/epochs)

        print("Episode {:d} reward score: {:0.2f}".format(i_episode, cumulated_reward))

    l = last_time_steps.tolist()
    l.sort()
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    # env.monitor.close()
    # gym.upload('/tmp/cartpole-experiment-1', algorithm_id='vmayoral simple Q-learning', api_key='your-key')