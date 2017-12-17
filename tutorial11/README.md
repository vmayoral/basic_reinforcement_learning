# A review of different AI techniques for RL

This tutorial will review the State Of The Art (SOTA) of RL using the [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures). Techniques
will get benchmarked using OpenAI gym-based environments.

### Index
- [Lesson 1](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial11#lesson-1-deep-rl-bootcamp-lecture-1-motivation--overview--exact-solution-methods)
- [Lesson 2](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial11#lesson-2-sampling-based-approximations-and-function-fitting)

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
	       <p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/583f0b49d413507672b398a769424fd2.svg?invert_in_darkmode" align=middle width=495.4223999999999pt height=35.479455pt/></p>
	     \right.
       $$
    - Perform a gradient descent on $(y_j - Q(\phi_j, a_j; \theta))^2$ with respect to the network parameters $\theta$
    - Every $C$ steps reset $\hat{Q} = Q$
  - **End For**
- **End For**
