from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import gym
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import argparse
from baselines import bench, logger

#parser
parser = argparse.ArgumentParser()
parser.add_argument('--environment', dest='environment', type=str, default='MountainCarContinuous-v0')
parser.add_argument('--num_timesteps', dest='num_timesteps', type=int, default=10000)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
args = parser.parse_args()

ncpu = 1
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)

tf.Session(config=config).__enter__()
def make_env():
    env = gym.make(str(args.environment))
    logger.configure("/tmp/experiments/"+str(args.environment)+"/PPO2/")
    env = bench.Monitor(env, logger.get_dir())
    # print(env.action_space.sample())
    return env

env = DummyVecEnv([make_env])
env = VecNormalize(env)

set_global_seeds(args.seed)
policy = MlpPolicy
ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
    lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
    ent_coef=0.0,
    lr=3e-4,
    cliprange=0.2,
    total_timesteps=args.num_timesteps,
    outdir="/tmp/experiments/"+str(args.environment)+"/PPO2/") # path for the log files (tensorboard) and models
