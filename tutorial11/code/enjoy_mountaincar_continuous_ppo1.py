import gym
import tensorflow as tf
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple

env = gym.make("MountainCarContinuous-v0")

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    hid_size=64, num_hid_layers=2)

# define the policy
pi = policy_fn('pi', env.observation_space, env.action_space)

# Define a TF session and restore graph
sess = U.make_session(num_cpu=1)
sess.__enter__()

# Load the previous trained graph
tf.train.Saver().restore(sess, '/tmp/experiments/continuous/PPO/models/TimeLimit_afterIter_80.model')
# tf.train.Saver().restore(sess, '/tmp/experiments/continuous/PPO/models/TimeLimit_afterIter_24.model')

env.render()
while True:
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        env.render()
        obs, rew, done, _ = env.step(pi.act(True, obs)[0])
        episode_rew += rew
    print("Episode reward", episode_rew)
