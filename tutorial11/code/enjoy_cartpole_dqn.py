import gym
from baselines import deepq

env = gym.make("CartPole-v0")
act = deepq.load("models/cartpole_model_DQN_[128, 128].pkl")

while True:
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        env.render()
        obs, rew, done, _ = env.step(act(obs[None])[0])
        episode_rew += rew
    print("Episode reward", episode_rew)
