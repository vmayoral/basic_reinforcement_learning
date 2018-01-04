from baselines.common import set_global_seeds
# from baselines.common.vec_env.vec_normalize import VecNormalize
# from baselines.ppo2 import ppo2
# from baselines.ppo2.policies import MlpPolicy
from baselines.deepqnaf import deepqnaf as naf
from baselines.deepqnaf.experiment import recursive_experiment
from baselines.deepqnaf.experiment import experiment
import gym
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import argparse

#parser
parser = argparse.ArgumentParser()
parser.add_argument('--environment', dest='environment', type=str, default='MountainCarContinuous-v0')
parser.add_argument('--max_episode_steps', dest='max_episode_steps', type=int, default=10000)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
args = parser.parse_args()

# create the environment
env = gym.make(str(args.environment))
initial_observation = env.reset()

# TODO: complete
