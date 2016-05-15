'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning

Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA

        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import gym
import numpy
import random
import pandas

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

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.monitor.start('/tmp/cartpole-experiment-1', force=True)
        # video_callable=lambda count: count % 10 == 0)

    goal_average_steps = 195
    max_number_of_steps = 200
    last_time_steps = numpy.ndarray(0)
    n_bins = 8
    n_bins_angle = 10

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
                    alpha=0.5, gamma=0.90, epsilon=0.1)

    for i_episode in xrange(3000):
        observation = env.reset()

        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation            
        state = build_state([to_bin(cart_position, cart_position_bins),
                         to_bin(pole_angle, pole_angle_bins),
                         to_bin(cart_velocity, cart_velocity_bins),
                         to_bin(angle_rate_of_change, angle_rate_bins)])

        for t in xrange(max_number_of_steps):	    	
            # env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation            
            nextState = build_state([to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)])

            # # If out of bounds
            # if (cart_position > 2.4 or cart_position < -2.4):
            #     reward = -200
            #     qlearn.learn(state, action, reward, nextState)
            #     print("Out of bounds, reseting")
            #     break

            if not(done):
                qlearn.learn(state, action, reward, nextState)
                state = nextState
            else:
                # Q-learn stuff
                reward = -200
                qlearn.learn(state, action, reward, nextState)
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                break    

    l = last_time_steps.tolist()
    l.sort()
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.monitor.close()
    # gym.upload('/tmp/cartpole-experiment-1', algorithm_id='vmayoral simple Q-learning', api_key='your-key')
