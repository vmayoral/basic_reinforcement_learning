import gym
from baselines.vpg import vpg
import tensorflow as tf
import argparse

#parser
parser = argparse.ArgumentParser()
# parser.add_argument('--graph', action='store_true')
# parser.add_argument('--render', action='store_true')
parser.add_argument('--environment', dest='environment', type=str, default='MountainCarContinuous-v0')
# parser.add_argument('--environment', dest='environment', nargs='+', type=str, default='InvertedPendulum-v1')
# parser.add_argument('--repeats', dest='repeats', type=int, default=1)
# parser.add_argument('--episodes', dest='episodes', type=int, default=10000)
parser.add_argument('--num_timesteps', dest='num_timesteps', type=int, default=10000)
# parser.add_argument('--train_steps', dest='train_steps', type=int, default=5)
# parser.add_argument('--learning_rate', dest='learning_rate', type=float, nargs='+', default=0.01)
# parser.add_argument('--batch_normalize', dest='batch_normalize', type=bool, default=True)
# parser.add_argument('--gamma', dest='gamma', type=float,nargs='+', default=0.99)
# parser.add_argument('--tau', dest='tau', type=float,nargs='+', default=0.99)
# parser.add_argument('--epsilon', dest='epsilon', type=float, nargs='+', default=0.1)
# parser.add_argument('--hidden_size', dest='hidden_size', type=int, nargs='+', default=32)
# parser.add_argument('--hidden_n', dest='hidden_n', type=int,nargs='+', default=2)
# parser.add_argument('--hidden_activation', dest='hidden_activation', nargs='+', default=tf.nn.relu)
# parser.add_argument('--batch_size', dest='batch_size', type=int, nargs='+', default=128)
# parser.add_argument('--memory_capacity', dest='memory_capacity', type=int, nargs='+', default=10000)
# parser.add_argument('-v', action='count', default=0)
# parser.add_argument('--load', dest='load_path', type=str, default=None)
# parser.add_argument('--output', dest='output_path', type=str, default=None)
# parser.add_argument('--covariance', dest='covariance', type=str, nargs='+', default="original")
# parser.add_argument('--solve_threshold', dest='solve_threshold', type=float, nargs='+', default=None) #threshold for having solved environment
args = parser.parse_args()

env = gym.envs.make(str(args.environment))
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
    policy_estimator = vpg.learn(env, policy_estimator, value_estimator,
                    max_timesteps=args.num_timesteps,
                    discount_factor=0.98,
                    print_freq=10,
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
