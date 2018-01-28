# A review of different AI techniques for RL

This tutorial will review the State Of The Art (SOTA) of RL using the [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures). Techniques
will get benchmarked using OpenAI gym-based environments.

### Index
- [Lesson 1](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial11#lesson-1-deep-rl-bootcamp-lecture-1-motivation--overview--exact-solution-methods)
- [Lesson 2](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial11#lesson-2-sampling-based-approximations-and-function-fitting)
- [Lesson 3](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial11#lesson-3-deep-q-networks)
- [Lesson 4A](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial11#lesson-4a-policy-gradients)
- [Lesson 4B](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial11#lesson-4b-policy-gradients-revisited)
- [Lesson 5](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial11#lesson-5-natural-policy-gradients-trpo-ppo)
- [Lesson 6](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial11#lesson-6-nuts-and-bolts-of-deep-rl-experimentation)

- [Extra Lesson: Frontiers Lecture I](#extra-lesson-frontiers-lecture-i)
- [Extra Lesson: Frontiers Lecture II](#extra-lesson-frontiers-lecture-ii)


### Lesson 1: Deep RL Bootcamp Lecture 1: Motivation + Overview + Exact Solution Methods
#### Notes from lesson
([video](https://www.youtube.com/watch?v=qaMdN6LS9rA))

Lesson treats how to solve MDPs using exact methods:
- Value iteration
- Policy iteration

Limitations are:
- Iteration over/storage for all states, actions and rewards require **small and discrete
state-action space**
- Update equations require access to the dynamical model (of the robot or whatever the agent is)

<img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/83298bfbf00ad918af5dfff006b94b06.svg?invert_in_darkmode" align=middle width=74.390085pt height=24.65759999999998pt/> expected utility/reward/value starting in <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/6f9bad7347b91ceebebd3ad7e6f6f2d1.svg?invert_in_darkmode" align=middle width=7.705549500000004pt height=14.155350000000013pt/>, taking action <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.689230000000004pt height=14.155350000000013pt/> and (thereafter)
acting optimally.


This Q-value function satisfies the Bellman Equation:
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/97e195d5485ae9fd519f1a97a1aebfb5.svg?invert_in_darkmode" align=middle width=396.96854999999994pt height=36.895155pt/></p>

For solving <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/e6f426deb3194d674706c8c9ea55188c.svg?invert_in_darkmode" align=middle width=19.730700000000002pt height=22.638659999999973pt/>, an interative method named "Q-value iteration" is proposed.

<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/551f01f2e1c0c044d42a3c14f3628968.svg?invert_in_darkmode" align=middle width=414.6747pt height=36.895155pt/></p>

Very simple, initial estimate all to 0. Within each iteration we will replace <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/e6f426deb3194d674706c8c9ea55188c.svg?invert_in_darkmode" align=middle width=19.730700000000002pt height=22.638659999999973pt/> by the current estimate of <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/964fc254befe9910f5b411f7026ba4d7.svg?invert_in_darkmode" align=middle width=20.261505000000003pt height=22.46574pt/> and compute <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/3b6d5dd34b842494d5be0641ecad1972.svg?invert_in_darkmode" align=middle width=36.905385pt height=22.46574pt/>.

##### Policy evaluation (and iteration)
Improving the policy over the time.
- Remember, value iteration:
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/08e682bfc54ea33c79102ef2b2640411.svg?invert_in_darkmode" align=middle width=371.03715pt height=36.895155pt/></p>


- Policy evaluation for a given <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/123d4737e41a5b18fb3abc6cc33a2451.svg?invert_in_darkmode" align=middle width=30.45108pt height=24.65759999999998pt/>:
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/d31217d8738082ebe26d7145bf60cbd2.svg?invert_in_darkmode" align=middle width=382.3248pt height=36.895155pt/></p>

We don't get to choose the action, it's frozen to <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/123d4737e41a5b18fb3abc6cc33a2451.svg?invert_in_darkmode" align=middle width=30.45108pt height=24.65759999999998pt/>. We compute the value of the policy <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/f30fdded685c83b0e7b446aa9c9aa120.svg?invert_in_darkmode" align=middle width=9.960225000000003pt height=14.155350000000013pt/>. We are now in a different MDP where every state, the action has been chosen for you. There's no choice.

### Lesson 2: Sampling-based Approximations and Function Fitting
#### Notes from lesson
([video](https://www.youtube.com/watch?v=qO-HUo0LsO4))

Given the limitations of the previous lesson we'll react as follows:
Limitations are:
- Iteration over/storage for all states, actions and rewards require **small and discrete
state-action space** -> **sampling-based approximations**
- Update equations require access to the dynamical model (of the robot or whatever the agent is) -> **Q/V function fitting methods**.


Given Q-value iteration as:
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/551f01f2e1c0c044d42a3c14f3628968.svg?invert_in_darkmode" align=middle width=414.6747pt height=36.895155pt/></p>

This equation is pretty expensive from a computational perspective. Consider a robot, you'll need to know each one of the possible future states and compute the corresponding probability of reaching those states.

Rewrite it as an expectation:
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/23944b1b0bdae05cf9955ac6b950b7ac.svg?invert_in_darkmode" align=middle width=391.37174999999996pt height=23.77353pt/></p>

This results in an algorithm called "tabular Q-learning" whose algorithm follows:

##### Algorithm: Tabular Q-Learning

- Start with <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/61037e77a80d0985cf3ec0676f94cf69.svg?invert_in_darkmode" align=middle width=56.855865pt height=24.65759999999998pt/> for all <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/6f9bad7347b91ceebebd3ad7e6f6f2d1.svg?invert_in_darkmode" align=middle width=7.705549500000004pt height=14.155350000000013pt/>, <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.689230000000004pt height=14.155350000000013pt/>.
- Get initial state <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/6f9bad7347b91ceebebd3ad7e6f6f2d1.svg?invert_in_darkmode" align=middle width=7.705549500000004pt height=14.155350000000013pt/>
- For <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/a758aaa74fe3ad6eb3e6dfb5d1dac4e7.svg?invert_in_darkmode" align=middle width=75.74193pt height=22.831379999999992pt/> till convergence:
  - Sample action <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.689230000000004pt height=14.155350000000013pt/>, get next state <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/675c2f5707a1fa7050c12adc1872ba32.svg?invert_in_darkmode" align=middle width=11.495550000000003pt height=24.716340000000006pt/>
  - If <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/675c2f5707a1fa7050c12adc1872ba32.svg?invert_in_darkmode" align=middle width=11.495550000000003pt height=24.716340000000006pt/> is terminal:
    - $target = R(s,a,s')$
    - Sample new initial state $s'$
  - else:
    - $target = R(s,a,s') + \gamma \underset{a'}{max} Q_k (s',a')$
  - <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/9d8af80ca3acefb0a628b4a0c227acf6.svg?invert_in_darkmode" align=middle width=305.216505pt height=24.65759999999998pt/>
  - <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/320bcc67abc355b59c9bd9b9edd7ee55.svg?invert_in_darkmode" align=middle width=44.771595000000005pt height=24.716340000000006pt/>


During the inital phases of learning, choosing greedily isn't optimal. If you only choose actions greedily you are restricting yourself to not explore alternative strategies.

Instead, what's used is <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/c7bd2b68b4bf6fc5be44f25166648272.svg?invert_in_darkmode" align=middle width=80.074665pt height=22.831379999999992pt/>.

Q-learning converges to optimal policy -- even if you're acting suboptimally!. This is called off-policy learning. Caveats:
- You have to explore enough
- You have to eventually make the learning rate small enough
- ... but not decrease it too fast, otherwise it will converge to wrong values.

The reinforcement learning performed by human race is a collective learning practice where we're not just looking at the experience of ourselves but also looking at the experience of other human beings.

Getting tabular Q-learning to work is unfeasible for most environments thereby "Approximate Q-Learning" is introduced. This typically gets represented through a parametrized Q-function: <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/8aab562ffcd233acb4b71ca480fdc3fd.svg?invert_in_darkmode" align=middle width=56.91856500000001pt height=24.65759999999998pt/> that can get represented with linear functions or with more complicated neural nets.

In some case, "Tabular Q-learning" is a special case of "Approximate Q-learning".

### Lesson 3: Deep Q-Networks
#### Notes from lesson
([video](https://www.youtube.com/watch?v=fevMOp5TDQs&t=3s))

The whole idea behind DQN is to make Q-learning look like it's supervised learning.

Two ideas to address the correlation between updates:
- Experience replay -> better data efficiency
- Use older set of weights to compute the targets (target network). This is good because if we fix this target weights then basically we have a fixed network that's producing our estimates and we have also those experience replay buffers, something like a dataset which we're sampling data from. So what we do with this is that we feed this Q values to something that almost looks like a fixed set of targets.

When one runs DQN in a new problem it might <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/43e7e5d805ae33e683f6ddfec02b6efd.svg?invert_in_darkmode" align=middle width=22.9911pt height=26.76201000000001pt/> transitions to get some reasonable results.

##### Algorithm: Deep Q-Learning (DQN)
- Initialize replay memory <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/78ec2b7008296ce0561cf83393cb746d.svg?invert_in_darkmode" align=middle width=14.066250000000002pt height=22.46574pt/> to capacity <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.999985000000004pt height=22.46574pt/>
- Initialize action-value function Q with random weights <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.173588500000005pt height=22.831379999999992pt/>
- Initialize target action-value function \hat{Q} with weights <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/5796af6b4874e982d27e2266fc65b534.svg?invert_in_darkmode" align=middle width=38.26482pt height=31.50708000000001pt/>
- **For** <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/36ce91b4d1d5f7f6f090d39a4bcf7571.svg?invert_in_darkmode" align=middle width=108.65414999999999pt height=22.831379999999992pt/> **do**:
  - Initialize sequence <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/4f74f0385042e12b8901d4278c3c23de.svg?invert_in_darkmode" align=middle width=52.9452pt height=14.155350000000013pt/> and preprocessed sequence <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/e3e4b0ecfb9ecde9eac8cb93b8461313.svg?invert_in_darkmode" align=middle width=76.74661499999999pt height=24.65759999999998pt/>
  - **For** <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/e5f2e56ef7cd46d1e431ff999386ad17.svg?invert_in_darkmode" align=middle width=55.268235000000004pt height=22.46574pt/> **do**
    - With probability $\epsilon$ select a random action $a_t$
    - otherwise select $a_t = argmax_a Q(\phi(s_t),a;\theta)$
    - Execute action $a_t$ in emulator and observe reward $r$, and image $x_{t+1}$
    - Set $s_{t+1} = s, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$
    - Store transition ($\phi_t, a_t, r_t, \phi_{t+1}$) in $D$
    - Sample random minibatch of transition (($\phi_j, a_j, r_j, \phi_{j+1}$))
    - Set $$
    y_j = \left\{
	       <p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/5c65f32daf00bd77b7f3c42b5542bdce.svg?invert_in_darkmode" align=middle width=431.88585pt height=37.78995pt/></p>
	     \right.
       $$
    - Perform a gradient descent on $(y_j - Q(\phi_j, a_j; \theta))^2$ with respect to the network parameters $\theta$
    - Every $C$ steps reset $\hat{Q} = Q$
  - **End For**
- **End For**

Preprocessed <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/f50853d41be7d55874e952eb0d80c53e.svg?invert_in_darkmode" align=middle width=9.794565000000006pt height=22.831379999999992pt/> elements correspond to 4 frames of the game stacked together as an input to the network
representing the state (this worked for them).

Value-based methods tend to be more robust parameter-wise. Much more than policy gradient methods. People in DeepMind is running algorithms with barely no variations on the hyperparams.

Double DQN, an upgrade of DQN. It exploits the fact that you have two networks: the online network and the target network. The idea is to use your online network to select the best action but then you use the target network to get the value estimate. You separate the <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/c329b295e6d71511af4f29af596de771.svg?invert_in_darkmode" align=middle width=57.50976000000001pt height=14.155350000000013pt/> from selecting the value.

Dueling DQN, make two separate channels:
- a channel that output a single channel, the value
- and another channel that outputs one number per action, the advantage
Summing this up (to get the output estimate of the Q-value) it will work much better in practice.

You can go beyond <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/c7bd2b68b4bf6fc5be44f25166648272.svg?invert_in_darkmode" align=middle width=80.074665pt height=22.831379999999992pt/> exploration, one good way of doing exploration with NNs is to add noise to the parameters of the NNs.

### Lesson 4A: Policy Gradients
#### Notes from lesson
([video](https://www.youtube.com/watch?v=S_gwYj1Q-44))

Learning `Q` or `V` can be really complicated. E.g. in a robotic grasp.
- Q implies that continuous or high state actions spaces are tricky. Reason:  argmax computation is tricky. DQN has an ouput per each possible action but how can we deal with it if we have a continuous set of actions?.

DQN, one outpuot per action. Highest score. In the continuous case, we can't have an output per action. So we then need to take the action as an input. *Input* is action and state, *output* is the value of that action/state combination. We need to solve a difficult optimization problem.

Methods described will be on-policy. Dynamic programming methods (e.g. DQN) explore better and are more sample efficient.

**Typically Policy optimization is *easier to get it to work*.**

The derivation done shows that it's valid even when the reward function is discontinuous or even unknown.

The sign of the reward seems to play a relevant role in policy gradient methods. Since the gradient:
- increase probability of paths with positive reward R.
- decrease probability of paths with negative reward R.

A baseline `b` is introduced to improve the formulation. Such baseline doesn't affect
while it doesn't depend on the action. The value function <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/4b196f4ff6569c417dbeb3d2fa4a6f4c.svg?invert_in_darkmode" align=middle width=17.689155000000003pt height=22.46574pt/> tells you how much reward you'll get.

Explanation at https://youtu.be/S_gwYj1Q-44?t=36m:
##### Algorithm: Vanilla Policy Gradient [[William, 1992](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)]
- Initialize policy (e.g. NNs) parameter <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.173588500000005pt height=22.831379999999992pt/> and baseline <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode" align=middle width=7.054855500000005pt height=22.831379999999992pt/>
- **for** <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/d0ab356ea9a56a11794552df3bdc04a6.svg?invert_in_darkmode" align=middle width=131.91634499999998pt height=21.683310000000006pt/> **do**
    - Collect a set of trajectories by executing the current policy
    - At each timestep in each trajectory, compute
      the return $R_t = \sum_{t'=t}^{T-1} \gamma^{t'-t}r_{t'}$, and
      the advantage estimate $\hat{A_t} = R_t - b(s_t)$.
    - Re-fit the baseline (recomputing the value function) by minimizing $|| b(s_t) - R_t||^2$, summed over all trajectories and timesteps.
    - Update the policy, using a policy gradient estimate $\hat{g}$,
      which is a sum of terms $\nabla_\theta log\pi(a_t | s_t,\theta)\hat(A_t)$
**end for**

Among the demonstrators shown, robot learns how to run in about 2000 episodes which according to them, could take 2 weeks of real training.

Simulation to real robot explained at https://youtu.be/S_gwYj1Q-44?t=50m20s. Simulating and transferring it into the real robot is an active research line. One interesting approach is to run experiments in a number of different randomized scenarios under the same policy (e.g.: **using different physics engines**).

Question asked about how to make a robot do specific things once it has learned basic behaviors (e.g.: walk). Answer was that problem had to be set up slightly differently. Instead of inputing to the NNs only states you would also input the direction of the goal. Paper "Hindsight Experience Replay" describes this method.

### Lesson 4B: Policy Gradients Revisited
#### Notes from lesson
([video](https://www.youtube.com/watch?v=tqrcjHuNdmQ))

Accounts for the total amount of parameters at https://youtu.be/tqrcjHuNdmQ?t=3m34s as follows:

```
# 80x80 pixel images
#input layer to (first and only) hidden layer
#  200  neurons in the hidden layer
80*80*200 (weights) + 200 (biases)
# hidden to output layer
200*1 (weights) + 1 (biases)
```

In total: <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/2029949e9314de1b33a7733ef73f8416.svg?invert_in_darkmode" align=middle width=393.99475499999994pt height=24.65759999999998pt/>.

There's no way to learn from static frames so what they do is concatenate frames together or use the difference.

Interesting way of putting Supervised Learning and Reinforcement Learning:

| Supervised Learning | Reinforcement Learning |
| --------------------|------------------------|
| We try to maximize: <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/3046efeaa042703a7db1fe0f2f07cef3.svg?invert_in_darkmode" align=middle width=101.21314499999998pt height=24.65792999999999pt/> for images <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode" align=middle width=14.045955000000003pt height=14.155350000000013pt/> and labels <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.710445000000004pt height=14.155350000000013pt/> | we have no lables so we sample <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/76ffd083102670368cf1acdadaa9598a.svg?invert_in_darkmode" align=middle width=83.70251999999999pt height=24.65759999999998pt/> |
| | once we collect a batch of rollouts, we maximize: <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/341bce2e1610551ae507502c9b8ee94f.svg?invert_in_darkmode" align=middle width=130.886745pt height=24.65792999999999pt/> |

where <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/4ebf880807deff5796460f39aea46f80.svg?invert_in_darkmode" align=middle width=16.979820000000004pt height=22.46574pt/> is called the advantage (-1,+1) depending on the result.

Discounting is heuristic to do a modulation of the blame for a bad (or good) reward. Typically represented as <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode" align=middle width=9.423975000000004pt height=14.155350000000013pt/>.

Simple implementation of a policy gradient using numpy. Code https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5 explained nicely starting at https://youtu.be/tqrcjHuNdmQ?t=23m19s. (good exercise would be to implement it using Tensorflow).

### Lesson 5: Natural Policy Gradients, TRPO, PPO
#### Notes from lesson
([video](https://www.youtube.com/watch?v=xvRrgxcpaHY))

Lesson about more advanced optimization methods to be used with Policy Gradients (PG).

Two limitations of "Vanilla Policy Gradient methods":
- hard to choose stepsizes (observation and reward distributions may change)
- sample efficiency (slow learning)

How to use all data available and compute the best policy:
- In Q-Learning we're optimizing the wrong objective. In Q-Learning you're trying to minimize some kind of Bellman error when what you care about is that your policy performs well.
- PG optimized the thing we care about but the downside is that they are not good at using all of the data we have available.

##### Algorithm: Trust Region Policy Optimization (TRPO)
For <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/afffe65a4a177f6f6cc5f27bb47d547d.svg?invert_in_darkmode" align=middle width=141.78318000000002pt height=21.683310000000006pt/> do
- Run policy for <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.889405000000002pt height=22.46574pt/> timesteps or <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.999985000000004pt height=22.46574pt/> trajectories
- Estimate advantage function at all timesteps:
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/a9edf5cff46266297cf927e217389deb.svg?invert_in_darkmode" align=middle width=227.49374999999998pt height=47.60745pt/></p>
   subject to $KL_{\pi_\theta_{old}}$ (\pi_theta) \le \delta$
End For

It appears that we can solve this constrained optimization problem efficiently by using conjugate gradient. This method is closely related to natural policy gradients, natural actor-critic and others.

Really theoretical. Summary of the theoretical methods at https://youtu.be/xvRrgxcpaHY?t=27m16s.

Limitations of TRPO:
- Hard to use with architectures with multiple outputs, e.g. like if the policy is outputing action probabilities and the value function then it's not clear what you should use instead of the KL divergence.
- Empirically it performs poorly on tasks requiring deep CNNs and RNNs
- Conjugate Gradient makes the implementation less flexible (a little bit more complicated)

Two alternatives to TRPO:
- KFAC: do blockwise approximation to Fisher Information Matrix (FIM)
- ACKTR: KFAC idea into A2C.
- PPO: use penalty instead of the constraint.

##### Algorithm: Proximal Policy Optimization (PPO)
For <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/d0ab356ea9a56a11794552df3bdc04a6.svg?invert_in_darkmode" align=middle width=131.91634499999998pt height=21.683310000000006pt/> do
- Run policy for <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.889405000000002pt height=22.46574pt/> timesteps or <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.999985000000004pt height=22.46574pt/> trajectories
- Estimate advantage function at all timesteps
- DO SGD on <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/5ce74f84a7c356ca02480a923de2b1fa.svg?invert_in_darkmode" align=middle width=69.080055pt height=27.656969999999987pt/> objective for some number of epochs
End For

It's:
- Much better for continuous control than TRPO, much better than atari (robotics).
- Compatible with RNNs and multiple output networks.

### Lesson 6: Nuts and Bolts of Deep RL Experimentation
Talk about tips on how to make decision on RL.

#### Aproaching new problems
Quick tips for new algorithms:
- Small environment
- Visualize everything
- If working with Hierarchical RL, we should have a way to identify what's the algorithm doing.
- Don't over optimize things when working with toy problems
- **Intuition**: Find out your own set of problems that you know very well and that allow you
to determine if an algorithm is learning or not.

Quick tips for a new task:
- Make things easier
- **Provide good input features**. E.g.: start "understanding the problem by" switching to "x-y" coordinates rather than start with full RGB images.
- **shape the reward function**: come up with a reward function that gives you fast feedback of whether you're doing the right thing or not.

POMDP design tips:
- Visualize random policies when training, if it does "sort-of-ok" work, then RL is likely the right thing.
- Make sure everything (obs and rewards mainly) is reasonable scaled. Everything with mean 0 and std deviation of 1.

Run your baselines:
- default parameters are useless
- Recommendation:
  - Cross-entropy method ([Wikipedia](https://en.wikipedia.org/wiki/Cross-entropy_method), [Gist implementation](https://gist.github.com/andrewliao11/d52125b52f76a4af73433e1cf8405a8f))
  - Well-tuned policy gradient method (e.g.: PPO)
  - Well-tuned Q-learning + SARSA method

Usually things work better when you have more samples (per batch). Bigger batches are recommended.

#### Ongoing development and tuning
Explore sensitivity to each parameter. If it's too sensitive, got lucky but not robust.

Have a system for continually benchmarking your code. Run a battery of benchmarks occasionally. CI? Automate your experiments.

#### General tuning strategies for RL

Standardizing data:
- Rescale observations with: <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/c9114abdbfc3b235fc212db86ca34b92.svg?invert_in_darkmode" align=middle width=204.72490499999995pt height=24.716340000000006pt/>
- Rescale rewards but don't shift the mean as that affects agents will to live
- Standardize prediction targets *is more complicated*.

Generally important parameters:
- Watch out with <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode" align=middle width=9.423975000000004pt height=14.155350000000013pt/>, it can ignore rewards delayed by x timesteps
- Action frequency, make sure it's human solvable.

General RL diagnostics:
- Look at episode returns min/max and stdev along with mean.
- Look at episode lengths (*sometimes more informative that the reward*), sometimes provides additional information (e.g.: solving problem faster, losing game slower)

#### Policy gradient strategies

Entropy as a diagnostic:
- If your entropy is going down really fast it means the policy is becoming deterministic and it's not going to explore anything. Be careful also if it's not going down (totally random always). **How do you measure entropy?** For most policies you can compute the entropy analytically. For continuous gaussian policies you can compute the differencial entropies.
- Use entropy bonus

Baseline explained variance:
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/57acf3528e91cb102517639a628ad46d.svg?invert_in_darkmode" align=middle width=469.9183499999999pt height=38.834894999999996pt/></p>

Policy initialization is relevant. Specially in supervised learning. Zero or tiny final layer, to maximize entropy.

#### Q-Learning strategies
- optimize memory usage (replay buffer)
- learning rate,
- schedules are often useful (<img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/c7bd2b68b4bf6fc5be44f25166648272.svg?invert_in_darkmode" align=middle width=80.074665pt height=22.831379999999992pt/>)
- converges slowly and has a **misterious warmup** period (DQN)

#### Miscellaneous advice
- Read older textbooks and theses
- Don't get too stucked on problems
- DQN performs pretty poorly for continuous control
- Techniques from supervised learning don't necessarily work in RL (e.g.: batch norm, dropout, big networks)

#### Questions
- **How long do you wait until you decide if the algorithm works or not?** No straight answer. For PG, it typically learns fast.
- **Do you use unit tests?** Only for particular mathematical things.
- **Do you have guidelines on how to much the algorithm to a particular task?** People have found that PG methods are probably the way to go if you don't care about sample complexity. PG is probably the safest way to go. DQN is a bit indirect of what's doing. If you care about sample complexity or need off-policy data, then DQN. Sample complexity is relevant if your simulator is expensive.
People have found that DQN works well with images as inputs while PG methods work better in continuous control tasks but it might be a historical accident.
- **Recommendations on textbooks**:
   - [Optimal control and dynamic programming](http://www.athenasc.com/dpbook.html)
   - ... (didn't catch them)
- **Comments on evolution strategies**: Lots of PG methods, some complicated. Evolution strategies (simple algorithm) as opposed to PG. OpenAI (and others) claimed EA work as well as PG. His opinion is that the sample complexity is worse by a constant factor (1,3, 100?). This might change between problems. EA might be a good alternative for problems with time dependencies.
- **Framework for hyperparameter optimization**: He uses uniform sampling. Works really well.


### Extra Lesson: Frontiers Lecture I
Two papers of interest for robotics regarding the SOTA:
- https://arxiv.org/abs/1610.04286
- https://arxiv.org/abs/1703.06907

#### Distributional RL
The value function gives the expected future discount return.

The Bellman equation in a slightly different form can be written as:
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/3d8ffa401194f5dd501f31d670397efd.svg?invert_in_darkmode" align=middle width=272.76644999999996pt height=17.289525pt/></p>

<img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/6cc7c111e4cb13cdefe9189eff1df02d.svg?invert_in_darkmode" align=middle width=48.883230000000005pt height=24.65759999999998pt/> will represent the distribution of the return of taking action <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.689230000000004pt height=14.155350000000013pt/> in state <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/6f9bad7347b91ceebebd3ad7e6f6f2d1.svg?invert_in_darkmode" align=middle width=7.705549500000004pt height=14.155350000000013pt/>. The distribution of the Bellman equation can be expressed as:
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/16862aac8de6ddad99cb3038bd6e07cc.svg?invert_in_darkmode" align=middle width=237.42345pt height=17.289525pt/></p>

(*capital letters are representing random variables*)

C51 is a specific instantiation of using this previous idea of distributions:
- C51 outputs a 51-way softmax for each Q-value.
- Bin the return into 51 equally-sized bins between -10 and 10
- The Q-value update becomes categorial (instead of the usual regression)

Pseudocode: Categorical Algorithm
- input A transition <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/23776aad854f2d33e83e4f4cad44e1b9.svg?invert_in_darkmode" align=middle width=14.360775000000002pt height=14.155350000000013pt/>, <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/9789555e5d8fa5de21171cc40c86d2cd.svg?invert_in_darkmode" align=middle width=13.655070000000002pt height=14.155350000000013pt/>, <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/39522195af502cac4f9d3a41f3e0f2ca.svg?invert_in_darkmode" align=middle width=12.382095000000003pt height=14.155350000000013pt/>, <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/f158a1d9b3f2502f7d6b75a1f4cfd188.svg?invert_in_darkmode" align=middle width=31.00482pt height=14.155350000000013pt/>, <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/8062ab43096502933fcef299185ecec3.svg?invert_in_darkmode" align=middle width=67.26621pt height=24.65759999999998pt/>
  - <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/04711cfefe190f73bf126d25ef03f334.svg?invert_in_darkmode" align=middle width=213.11845499999998pt height=24.65792999999999pt/>
  - <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/13002125e23ccdf053bad55834ad7ef0.svg?invert_in_darkmode" align=middle width=155.31087pt height=24.65759999999998pt/>
  - <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/c76990ee30392ab418de29891d278910.svg?invert_in_darkmode" align=middle width=162.94360500000002pt height=22.46574pt/>
  - <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/e5d75b83bbf0fc758ab9d3de3206a463.svg?invert_in_darkmode" align=middle width=133.300035pt height=22.831379999999992pt/> do
      - # Compute the projection of $T_z_j$  onto the support ${z_i}$
      - $T_z_j \arrow [r_t + \gamma_t z_j]$_{V_{MIN}}^{V_{MAX}}

... (complete algorithm, minute 8:42)

Resuls shown show that C51 improve DQN by 700%. Furthermore, results using categorical classification (rather than regression) gets us good results. It was a huge advance.

Among the good things about Distributional RL, data efficiency is greatly improved. Why does it work so well? Don't really know. Is it learning to model uncertainty?

PG gradient might obtain better results for the different environments but the hyperparams need to be tuned for each environment.

#### Model-based RL
Big expectation about it. They think it'll be pretty important. Specially for cases where you have a simulator that's really expensive or you want to do something in the real world, like with robots.

#### Neural Episodic control (NEC)
An interesting new approach that includes the parametric approach (which is used in all typical methods) and a non-parametric one. Good results on the environments presented.

#### Hierarchical Reinforcement Learning (HRL)
Intuitive appealing. It hasn't landed into massive breakthroughs just yet.

The basic idea is to capture the structure in the policy space. Goes back to 1993 with "Feudal Reinforcement Learning".

#### Deep RL for Real Robots
It feels like in the next few years, learning in robots will get us to the point of having robots doing real cool stuff. All the methods they looked so far were really data innefficient.

He claimed we were close to breakthrough (or already on it).
Two main approaches:
- Learning in simulation and then doing transfer learning. Simulation variances to fill in the simulation gap.
- Imitation learning will be important.

### Extra Lesson: Frontiers Lecture II

Small changes in the observatoins lead to the robot failing to generalize.

> We can't build a system that doesn't make mistakes. But maybe we can learn from those mistakes.

What does it take for RL to learn from its mistakes in the real world?
- Focus on **diversity** rather **proficiency**
- Self-supervised algorithms, they must have access to the reward.
- Learn quickly and efficiently
- Fix mistakes when they happen, adaptability.

Current benchmarks in robotics do not follow the goals of images (e.g.: imagenet). They are small-scale, emphasize mastery and are evaluated on performance.

#### Can we self-supervise diverse real-world tasks?
By self-supervised learning with robots, Sergey means to have the robot learn a task by itself without human intervention (or with as little as possible).

The open loop (just looking once at the image) works much worse (33.7% failures) than the closed-loop (17.5% failures). There's a big benefit at doing continuous control.

Training with multiple robots improves even more the performance (minute 16:30). They first trained the whole network in one robot and then on the other robot. The end effector is the same, the rest different. Work by Julian Ibarz.

```
A few comments/ideas after watching this:
- Modular robots can extend their components seamlessly.
- This brings clear advantages for the construction of robots however training them with DRL becomes expensive due to the following reasons:
  - Every small change in the physical structure
of the robot will require a complete retrain.
  - Building up the tools to train modular robots (simulation
    model, virtual drivers for the different actuators, etc)
    becomes very expensive.
  - Transferring the results to the real robot is complex given the flexibility of these systems.

- We present a framework/method that makes use of traditional
tools in the robotics environment (e.g. ROS, Gazebo, etc.) which
 simplifies the process of building modular robots and their
 corresponding tools to be trained using DRL techniques. Besides
integration with the traditional robotics tools, our
framework/method includes baseline implementations for the most
common DRL techniques for both value and policy
 iteration methods. Using this framework we present the results obtained benchmarking DRL methods in a modular robot. Configurations with 3 and 4 degrees of freedom are presented while performing the same task. In addition, we present our insights about the impact of the simulation acceleration in the final reward and conclude with our hypotheses about


 Future work: a) generalization between different DOFs (refer to previous work from Julian Ibarz where the overall performance got better after training in two different robots), b) further work on transfer learning with simulation variance, c) inclusion of imitation learning to accelerate the process of learning a particular task, d) inclusion of images to get a much richer signal from where to learn.

Conclusions:
- Change titles to:
  - DRL methods/framework for modular robots
  - An information model for modular robots
- Participate in the Amazon Picking Challenge with a modular robot and DRL?
```

Semantic picking results presented. Pretty nice.

#### Where can we get goal supervision
(imitation learning?)

#### How can new tasks build on prior tasks
...

### Lesson 8: Derivative Free Methods
