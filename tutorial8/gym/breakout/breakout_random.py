'''
Random Breakout AI player

    @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import gym
import numpy
import random
import pandas

if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    env.monitor.start('/tmp/breakout-experiment-1', force=True)
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


    l = last_time_steps.tolist()
    l.sort()
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.monitor.close()
    # gym.upload('/tmp/cartpole-experiment-1', algorithm_id='vmayoral simple Q-learning', api_key='your-key')
