'''
Doom AI player using Deep Q-Learning

    @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import gym
import numpy
import random
import pandas

if __name__ == '__main__':
    env = gym.make('DoomBasic-v0')
    env.monitor.start('/tmp/doom-experiment-1', force=True)
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
