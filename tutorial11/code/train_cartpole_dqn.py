import gym
from baselines import deepq

def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

env = gym.make("CartPole-v0")
models = [[64], [64,64], [128,128], [256,256]]

for m in models:
    act = deepq.learn(
        env,
        q_func=deepq.models.mlp(m),
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        outdir="/tmp/experiments/DQN/"+str(m)
    )
print("Saving model to cartpole_model.pkl")
act.save("cartpole_model.pkl")
