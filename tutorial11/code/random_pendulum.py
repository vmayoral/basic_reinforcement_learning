import gym

env = gym.make("Pendulum-v0")
while True:
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        env.render()
        obs, rew, done, _ = env.step(env.action_space.sample())
        episode_rew += rew
    print("Episode reward", episode_rew)
