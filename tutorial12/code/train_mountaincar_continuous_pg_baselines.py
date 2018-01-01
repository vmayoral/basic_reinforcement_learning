import gym
from baselines.vpg import vpg
import tensorflow as tf

env = gym.envs.make("MountainCarContinuous-v0")
scaler, featurizer = vpg.preprocess(env)
tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = vpg.PolicyEstimator(env, learning_rate=0.001)
value_estimator = vpg.ValueEstimator(env, learning_rate=0.1)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # Note, due to randomness in the policy the number of episodes you need varies
    # g = tf.Graph()
    # with g.as_default():
    #     for i in range(10):
    policy_estimator = vpg.learn(env, policy_estimator, value_estimator, max_timesteps=10000,
                    discount_factor=0.95,
                    print_freq=1,
                    outdir="/tmp/experiments/continuous/VPG/other")

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
