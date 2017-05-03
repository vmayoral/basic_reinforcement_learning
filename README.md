Basic Reinforcement Learning (RL)
============================

This repository aims to provide an introduction series to reinforcement learning (RL) by delivering a walkthough on how to code different RL techniques.

### Background review 

Reinforcement Learning RL is a "a way of programming agents by reward and punishment without needing to specify how the task is to be achieved" [Kaelbling, Littman, & Moore, 96](https://www.cs.cmu.edu/~tom/10701_sp11/slides/Kaelbling.pdf).

The basic RL problem includes **states** (s), **actions** (a) and **rewards** (r). The typical formulation is as follows:
![captura de pantalla 2017-05-03 a las 14 58 28](https://cloud.githubusercontent.com/assets/1375246/25661451/febed5f2-3010-11e7-9867-451f0711b7e0.png)

The goal of RL is to select actions _a_ to move around states _s_  to maximize future reward _r_. Three key additional components in RL are:
- **policy** (π): the policy is the agent's behavior, in other words, how to select action (_a_) in certain state (_s_)
- **Value function** (V): prediction of the future reward or how much reward will I get from action _a_ in state _s_
- **Model**: representation of the environment, learnt from experience

#### Approaches to RL
- **Value-based RL**: Estimate the optimal value function Q∗(s, a). This is the maximum value achievable under any policy
- **Policy-based RL**: Search directly for the optimal policy π∗. This is the policy achieving maximum future reward
- **Model-based RL**: Build a model of the environment. Plan (e.g. by lookahead) using model

_There's also the actor-critic (e.g.: DDPG) techniques which learn both policies and value functions simultaneously._

#### Further classification of RL
- **Model-free or model-based**: From https://www.quora.com/What-is-the-difference-between-model-based-and-model-free-reinforcement-learning:
>Model based learning attempts to model the environment, and then based on that model, choose the most appropriate policy. Model-free learning attempts to learn the optimal policy in one step
- **On-policy or Off-policy**: From https://datascience.stackexchange.com/questions/13029/what-are-the-advantages-disadvantages-of-off-policy-rl-vs-on-policy-rl:

>   - On-policy methods:
>     - attempt to evaluate or improve the policy that is used to make decisions,
>     - often use soft action choice, i.e. π(s,a)>0,∀aπ(s,a)>0,∀a,
>     - commit to always exploring and try to find the best policy that still explores,
>     - may become trapped in local minima.
>   - Off-policy methods:
>     - evaluate one policy while following another, e.g. tries to evaluate the greedy policy while following a more exploratory scheme,
>     - the policy used for behaviour should be soft,
>     - policies may not be sufficiently similar,
>     - may be slower (only the part after the last exploration is reliable), but remains more flexible if alternative routes appear.

#### Based on:
- http://icml.cc/2016/tutorials/deep_rl_tutorial.pdf
- http://www.inf.ed.ac.uk/teaching/courses/rl/slides15/rl05.pdf
- http://www2.econ.iastate.edu/tesfatsi/RLUsersGuide.ICAC2005.pdf
- https://es.slideshare.net/DeepLearningJP2016/dlintroduction-of-reinforcement-learning

### Tutorials:
- Tutorial 1: [Q-learning](tutorial1/README.md)
- Tutorial 2: [SARSA](tutorial2/README.md)
- Tutorial 3: [Exploring OpenAI gym](tutorial3/README.md)
- Tutorial 4: [Q-learning in OpenAI gym](tutorial4/README.md)
- Tutorial 5: [Deep Q-learning (DQN)](tutorial5/README.md)
- Tutorial 6: [Deep Convolutional Q-learning](tutorial6/README.md)
- Tutorial 7: [Reinforcement Learning with ROS and Gazebo](tutorial7/README.md)
- Tutorial 8: [Reinforcement Learning in DOOM](tutorial8/README.md)

### References:
- Chris Watkins, Learning from Delayed Rewards, Cambridge, 1989 ([thesis](http://www.cs.rhul.ac.uk/home/chrisw/new_thesis.pdf))
- Awesome Reinforcement Learning repository, https://github.com/aikorea/awesome-rl
- Reinforcement learning CS9417ML, School of Computer Science & Engineering, UNSW Sydney, http://www.cse.unsw.edu.au/~cs9417ml/RL1/index.html
- Reinforcement learning blog posts, https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/
- OpenAI gym docs, https://gym.openai.com/docs 
- Vincent Bons implementations, https://gist.github.com/wingedsheep
- David Silver's Deep Reinforcement Learning talk, http://videolectures.net/rldm2015_silver_reinforcement_learning/
- Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). OpenAI Gym. arXiv preprint arXiv:1606.01540.
