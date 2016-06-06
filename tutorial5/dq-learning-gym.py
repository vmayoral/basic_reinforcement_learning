'''
DQN approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning

FIXME: 
        - ANN not performing at all

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
from keras.optimizers import RMSprop, SGD

monitor = False

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

def build_state(features):    
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

class DQN:
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
        # self.network.add(Dense(100, init='lecun_uniform', input_shape=(4,)))
        self.network.add(Dense(100, init='lecun_uniform', input_shape=(4,)))
        self.network.add(Activation('relu'))
        # self.network.add(Activation('tanh'))
        # self.network.add(Dropout(0.2))

        self.network.add(Dense(80, init='lecun_uniform'))
        self.network.add(Activation('relu'))
        # # self.network.add(Activation('tanh'))
        # # self.network.add(Dropout(0.2))

        self.network.add(Dense(2, init='lecun_uniform'))
        self.network.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        # rms = RMSprop()
        rms = SGD()
        self.network.compile(loss='mse', optimizer=rms)
        # Get a summary of the network
        self.network.summary()

    def learnQ(self, state, action, reward, newState, terminal=False):
        '''
        DQN learning:
            Instead of the Q-learning:
                Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
            we'll be updating the network following:
                target = reward(s,a) + gamma * max(Q(s')
        '''
        # oldv = self.q.get((state, action), None)
        # if oldv is None:
        #     self.q[(state, action)] = reward
        # else:
        #     self.q[(state, action)] = oldv + self.alpha * (value - oldv)

        state = numpy.asarray(state)
        state = state.reshape(1,4)

        newState = numpy.asarray(newState)
        newState = newState.reshape(1,4)
        
        qval = self.network.predict(state, batch_size=1)
        newQ = self.network.predict(newState, batch_size=1)
        # if (qval==newQ).all():
        #     pass
        # else:
        #     print("NOT EQUAL!")
        maxNewQ = numpy.max(newQ)
        
        y = numpy.zeros((1,2))
        y[:] = qval[:]
        if terminal:
            newReward = reward
        else:
            newReward = (reward + (self.gamma * maxNewQ))
        y[0][action] = newReward #target output        
        self.network.fit(state, y, batch_size=1, nb_epoch=1, verbose=0)

        # print("\tstate: "+str(state))
        # print("\tnewState: "+str(newState))
        # print("\taction: "+str(action))
        # print("\tqval: "+str(qval))
        # print("\tnewQval: "+str(newQ))
        # print("\treward: "+str(reward))
        # print("\tnewReward: "+str(newReward))
        # print("\ty: "+str(y))
        

    def chooseAction(self, state, return_q=False):        
        if (random.random() < self.epsilon): #choose random action
            action = numpy.random.randint(0,2)
        else: #choose best action from Q(s,a) values
            # convert to a numpy array
            state = numpy.asarray(state)
            state = state.reshape(1,4)
            # Let's run our Q function on state "state" to get Q values for all possible actions
            qvals = self.network.predict(state, batch_size=1)
            # Select the neuron that fired the most
            action = numpy.argmax(qvals)

            q = qvals[0][action]
            # if return_q: # if they want it, give it!
            #     return action, q        
        return action

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    if monitor:
        env.monitor.start('/tmp/cartpole-experiment-1', force=True)
        # video_callable=lambda count: count % 10 == 0)

    epochs = 500
    goal_average_steps = 195
    max_number_of_steps = 200
    last_time_steps = numpy.ndarray(0)

    # Discretization of the space
    n_bins = 10
    n_bins_angle = 10

    number_of_features = env.observation_space.shape[0]
    last_time_steps = numpy.ndarray(0)

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
    pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
    cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

    # The Deep Q-learn algorithm
    dqn = DQN(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.99)
    # The Q-learn algorithm
    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.1)


    for i_episode in xrange(epochs):
        observation = env.reset()

        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation            
        state = build_state([to_bin(cart_position, cart_position_bins),
                         to_bin(pole_angle, pole_angle_bins),
                         to_bin(cart_velocity, cart_velocity_bins),
                         to_bin(angle_rate_of_change, angle_rate_bins)])
        state_raw = [to_bin(cart_position, cart_position_bins),
                         to_bin(pole_angle, pole_angle_bins),
                         to_bin(cart_velocity, cart_velocity_bins),
                         to_bin(angle_rate_of_change, angle_rate_bins)]


        cumulated_reward = 0        
        for t in xrange(max_number_of_steps):           
            # env.render()

            # Pick an action based on the current state            
            action_qlearn = qlearn.chooseAction(state)
            action_dqn = dqn.chooseAction(state_raw)

            # print("\t\tdqn: "+str(action_dqn))
            # print("\t\tqlearn: "+str(action_qlearn))
            # action = action_qlearn
            action = action_dqn

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation            
            nextState = build_state([to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)])

            nextState_raw = [to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)]


            # # If out of bounds
            # if (cart_position > 2.4 or cart_position < -2.4):
            #     reward = -200
            #     dqn.learn(state, action, reward, nextState)
            #     print("Out of bounds, reseting")
            #     break

            if not(done):
                dqn.learnQ(state_raw, action, reward, nextState_raw)
                qlearn.learn(state, action, reward, nextState)
                state = nextState
                cumulated_reward += reward
            else:
                # Q-learn stuff
                reward = -200
                dqn.learnQ(state_raw, action, reward, nextState_raw, done)
                qlearn.learn(state, action, reward, nextState)
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                cumulated_reward += reward

                if dqn.epsilon > 0.1:
                    dqn.epsilon = dqn.epsilon - (1.0/epochs)
                # print(dqn.epsilon)
                break    

        print("Episode {:d} reward score: {:0.2f}".format(i_episode, cumulated_reward))

    l = last_time_steps.tolist()
    l.sort()
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    if monitor:
        env.monitor.close()
    # gym.upload('/tmp/cartpole-experiment-1', algorithm_id='vmayoral simple Q-learning', api_key='your-key')