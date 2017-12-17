import gym
import gym_cryptocurrencies
from baselines import deepq

def callback(lcl, glb):
    # stop training if reward exceeds 199
    # is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return False

# env = gym.make("CartPole-v0")
env = gym.make('simplified_trader-v0')
model = deepq.models.mlp([64])
act = deepq.learn(
    env,
    q_func=model,
    lr=1e-3,
    max_timesteps=100000,
    buffer_size=50000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    print_freq=10,
    callback=callback
)
print("Saving model to simplified_trader_dqn.pkl")
act.save("simplified_trader_dqn.pkl")
