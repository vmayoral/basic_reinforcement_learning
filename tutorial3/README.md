Basic Reinforcement Learning Tutorial 2: OpenAI gym
===================================================

This tutorial will cover the basics of the new (at the time of writting) OpenAI gym. A great tool that represents a common environment for programmers to challenge Reinforcement Learning (RL) algorithms. The tutorial will digest the [OpenAI gym docs](https://gym.openai.com/docs) by the insight obtained while going through the information and code available.

## Table of Contents
- [Installing](#installing)
- [The environment](#environment)
- [OpenAI gym spaces](#spaces)
- [Recording](#recording)
- [Uploading](#uploading)

<div id='installing'/>
### Installing OpenAI gym

```
git clone https://github.com/openai/gym
cd gym
pip install -e .[all]
# With Ubuntu 16.04 and latest Anaconda, the following dependencies still needed to install.
# xvfb libav-tools xorg-dev libsdl2-dev swig cmake
```

Installing OpenAI seems to be pretty straightforward with Python 2.X.

<div id='environment'/>
### The environment

The environment is the most relevant element of the gym. It abstracts each different scenario in an `env` object. According to their docs:

```
The (Env class is the) main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        reset
        step
        render
    When implementing an environment, override the following methods
    in your subclass:
        _step
        _reset
        _render
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality to over time.
```

Each `env` element seems contains a `step` function which takes an `action` as a parameter and returns 4 values:

- `observation` (object): an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
- `reward` (float): amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
- `done` (boolean): whether it's time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
- `info` (dict): diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment's last state change). 


Here's a simple use of such functions:
```python
import gym
env = gym.make('CartPole-v0')
for i_episode in xrange(20):
    observation = env.reset()
    for t in xrange(100):
        env.render()
        print observation
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break

```

To get a list of all the available environments within the OpenAI gym just do the following:
```python
from gym import envs
print envs.registry.all()
```
(**at the time of writing, `152`!**)

When looking at the code, environments all descend from the [Env](https://github.com/openai/gym/blob/master/gym/core.py#L10) base class. Let's have a quick look at the [CartPole environment's code](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py):

```python
class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.reset()
        self.viewer = None

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([self.x_threshold, np.inf, self.theta_threshold_radians * 2, np.inf])
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self.steps_beyond_done = None

        ...
```
Overall, the `CartPoleEnv` class is only **135 lines** of code which shows that there's a decent level of  abstractions introduced by OpenAI team.



<div id='spaces'/>
### OpenAI gym spaces

OpenAI gym spaces provide a classification state spaces and action spaces, so that users can write generic code that applies to any Environment.
    
Each environment describes the valid actions and observations through its `Space` class. Inspecting the `Space` of a particular environment looks like the following:
```python
import gym
env = gym.make('CartPole-v0')
print env.action_space
#> Discrete(2)
print env.observation_space
#> Box(4,)

print env.observation_space.high
#> array([ 2.4       ,         inf,  0.20943951,         inf])
print env.observation_space.low
#> array([-2.4       ,        -inf, -0.20943951,        -inf])
```

The spaces implemented are available at https://github.com/openai/gym/tree/master/gym/spaces.

<div id='recording'/>
### Recording

The OpenAI gym embeds easy to use recording capabilities which simplify the process of recording your algorithm's performance on an environment and filming these results. Here's the example provided:
```python
import gym
env = gym.make('CartPole-v0')
env.monitor.start('/tmp/cartpole-experiment-1')
for i_episode in xrange(20):
    observation = env.reset()
    for t in xrange(100):
        env.render()
        print observation
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break

env.monitor.close()
```

The code in charge of implementing this capability is available at [`monitor.py`](https://github.com/openai/gym/blob/master/gym/monitoring/monitor.py). A few interesting facts:
-  For finer-grained control over how often videos are collected, use the video_callable argument, e.g. `monitor.start(video_callable=lambda count: count % 100 == 0)` to record every 100 episodes. (`count` is how many episodes have completed in code)
- To deactivate video recording: `monitor.configure(video_callable=lambda count: False)`.
- Monitor supports multiple threads and multiple processes writing to the same directory of training data. The data will later be joined by scoreboard.upload_training_data and on the server.


<div id='uploading'/>
### Uploading

Results can easily be uploaded using:
```python
import gym
gym.upload('/tmp/cartpole-experiment-1', api_key='sk_5YJsWfHOQwOLiU3AAVyYeA')
```

The `upload` function is defined at [`scoreboard/api.py`](https://github.com/openai/gym/blob/master/gym/scoreboard/api.py#L18). Within its definition, there's a short description of the arguments that this function accepts:

```
    Args:
        training_dir (Optional[str]): A directory containing the results of a training run.
        algorithm_id (Optional[str]): An arbitrary string indicating the paricular version of the algorithm (including choices of parameters) you are running.
        writeup (Optional[str]): A Gist URL (of the form https://gist.github.com/<user>/<id>) containing your writeup for this evaluation.
        api_key (Optional[str]): Your OpenAI API key. Can also be provided as an environment variable (OPENAI_GYM_API_KEY).
```


