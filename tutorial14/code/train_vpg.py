import gym
from baselines.vpg import vpg
import tensorflow as tf
import argparse

#parser
parser = argparse.ArgumentParser()
parser.add_argument('--environment', dest='environment', type=str, default='MountainCarContinuous-v0')
parser.add_argument('--num_timesteps', dest='num_timesteps', type=int, default=10000)
args = parser.parse_args()

env = gym.envs.make(str(args.environment))
tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = vpg.PolicyEstimator(env, learning_rate=0.001)
value_estimator = vpg.ValueEstimator(env, learning_rate=0.1)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    policy_estimator = vpg.learn(env, policy_estimator, value_estimator,
                    max_timesteps=args.num_timesteps,
                    discount_factor=0.98,
                    print_freq=1,
                    outdir="/tmp/experiments/"+str(args.environment)+"/VPG/")

    # plotting.plot_episode_stats(stats, smoothing_window=10)

    # # Try it out
    # state = env.reset()
    # while 1:
    #     env.render()
    #     action = policy_estimator.predict(state)
    #     next_state, reward, done, _ = env.step(action)
    #     if done:
    #         state = env.reset()
    #     state = next_state
