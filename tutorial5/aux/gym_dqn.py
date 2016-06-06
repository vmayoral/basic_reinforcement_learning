import sys
import gym
from dqn import Agent

num_episodes = 20

env_name = sys.argv[1] if len(sys.argv) > 1 else "MsPacman-v0"
env = gym.make(env_name)
env.monitor.start('/tmp/pacman-experiment-1', force=True)

agent = Agent(state_size=env.observation_space.shape,
              number_of_actions=env.action_space.n,
              save_name=env_name)

for e in xrange(num_episodes):
    observation = env.reset()
    done = False
    agent.new_episode()
    total_cost = 0.0
    total_reward = 0.0
    frame = 0
    while not done:
        frame += 1
        #env.render()
        action, values = agent.act(observation)
        #action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_cost += agent.observe(reward)
        total_reward += reward
    print("total reward "+ str(total_reward))
    print("mean cost " +str(total_cost/frame))

env.monitor.close()