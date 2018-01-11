from baselines.deepqnaf.learn import learn
import gym
import tensorflow as tf
import argparse

#parser
parser = argparse.ArgumentParser()
parser.add_argument('--environment', dest='environment', type=str, default='MountainCarContinuous-v0')
parser.add_argument('--num_timesteps', dest='num_timesteps', type=int, default=100000)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
args = parser.parse_args()

with tf.Session() as sess:
    # create the environment
    env = gym.make(str(args.environment))

    # Check continuity of the environment
    assert isinstance(env.observation_space, gym.spaces.Box), \
        "observation space must be continuous"
    assert isinstance(env.action_space, gym.spaces.Box), \
        "action space must be continuous"

    # Fix these two values and calculate episodes from the given timesteps
    max_steps = 200
    update_repeat = 5
    max_episodes = args.num_timesteps // (max_steps*update_repeat)

    learn (env, sess,
            hidden_dims=[64,64],
            # hidden_dims=[100,100],
            use_batch_norm=True,
            # learning_rate=0.001,
            learning_rate=0.0001,
            batch_size=100, # kind of like the size of the replay buffer
            max_steps=max_steps,
            update_repeat=update_repeat,
            max_episodes=max_episodes,
            outdir="/tmp/experiments/"+str(args.environment)+"/DEEPQNAF/")
