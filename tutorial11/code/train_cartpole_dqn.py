import gym
from baselines import deepq
import tensorflow as tf
# from baselines import bench, logger
# import os

def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


# Define the environment
env = gym.make("CartPole-v0")

# # set up the logger
# logdir = '/tmp/experiments/discrete/DQN/'
# logger.configure(os.path.abspath(logdir))
# print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))

# models = [[64], [64,64], [128,128], [256,256]]
models = [[64], [128], [64,64], [128,128], [256,256]]

for m in models:
    g = tf.Graph()
    with g.as_default():
        # tf.reset_default_graph()
        act = deepq.learn(
            env,
            q_func=deepq.models.mlp(m),
            lr=1e-3,
            max_timesteps=10000,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            print_freq=10,
            callback=callback,
            outdir="/tmp/experiments/discrete/DQN/"+str(m)
        )
        act.save("models/cartpole_model_DQN_"+str(m)+".pkl")

# print("Saving model to cartpole_model.pkl")
# act.save("cartpole_model.pkl")
