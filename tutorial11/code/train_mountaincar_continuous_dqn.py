import gym
from baselines import deepq

# env = gym.make("MountainCar-v0")
env = gym.make("MountainCarContinuous-v0")
print(env.action_space.shape)
# Enabling layer_norm here is import for parameter space noise!
model = deepq.models.mlp([64], layer_norm=True)
act = deepq.learn(
    env,
    q_func=model,
    lr=1e-3,
    max_timesteps=100000,
    buffer_size=50000,
    exploration_fraction=0.1,
    exploration_final_eps=0.1,
    print_freq=10,
    param_noise=False
)
print("Saving model to mountaincar_model.pkl")
act.save("mountaincar_continuous_model.pkl")
