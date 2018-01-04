import gym
from baselines.ppo1 import mlp_policy, pposgd_simple
import tensorflow as tf
import argparse

#parser
parser = argparse.ArgumentParser()
parser.add_argument('--environment', dest='environment', type=str, default='MountainCarContinuous-v0')
parser.add_argument('--num_timesteps', dest='num_timesteps', type=int, default=10000)
args = parser.parse_args()

env = gym.envs.make(str(args.environment))

g = tf.Graph()
with g.as_default():
    # tf.reset_default_graph()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    pposgd_simple.learn(env,
            policy_fn,
            max_timesteps=args.num_timesteps,
            timesteps_per_actorbatch=2048, # timesteps per actor per update
            # timesteps_per_actorbatch=128, # timesteps per actor per update
            clip_param=0.2,
            entcoeff=0.0,
            optim_epochs=10,
            optim_stepsize=3e-4,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule='linear',
            save_model_with_prefix=str(env.__class__.__name__), # typically, the env.
            outdir="/tmp/experiments/"+str(args.environment)+"/PPO1/") # path for the log files (tensorboard) and models


    # act.save("models/mountaincar_continuous_model_PPO_"+str(m)+".pkl")
