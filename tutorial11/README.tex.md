# A review of different AI techniques for RL

This tutorial will review the State Of The Art (SOTA) of RL using the [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures). Techniques
will get benchmarked using OpenAI gym-based environments.

### Index
- [Lesson 1](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial11#lesson-1-deep-rl-bootcamp-lecture-1-motivation--overview--exact-solution-methods)
- [Lesson 2](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial11#lesson-2-sampling-based-approximations-and-function-fitting)
- [Lesson 3](https://github.com/vmayoral/basic_reinforcement_learning/tree/master/tutorial11#lesson-3-deep-q-networks)

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

$Q^*(s,a) = $ expected utility/reward/value starting in $s$, taking action $a$ and (thereafter)
acting optimally.


This Q-value function satisfies the Bellman Equation:
$$
Q^*(s,a) \leftarrow  \sum_{s'} P(s' | s,a) \cdot (R(s,a,s') + \gamma \underset{a'}{max} Q^* (s',a') )
$$

For solving $Q^*$, an interative method named "Q-value iteration" is proposed.

$$
Q_{k+1}(s,a) \leftarrow  \sum_{s'} P(s' | s,a) \cdot (R(s,a,s') + \gamma \underset{a'}{max} Q_k (s',a') )
$$

Very simple, initial estimate all to 0. Within each iteration we will replace $Q^*$ by the current estimate of $Q_k$ and compute $Q_{k+1}$.

##### Policy evaluation (and iteration)
Improving the policy over the time.
- Remember, value iteration:
$$
V^*_k(s) \leftarrow   \underset{max}{a} \sum_{s'} P(s' | s,a) \cdot (R(s,a,s') + \gamma V^*_{k-1} (s') )
$$


- Policy evaluation for a given $\pi(s)$:
$$
V^\pi_k(s) \leftarrow  \sum_{s'} P(s' | s,\pi(s)) \cdot (R(s,\pi(s),s') + \gamma V^\pi_{k-1} (s) )
$$

We don't get to choose the action, it's frozen to $\pi(s)$. We compute the value of the policy $\pi$. We are now in a different MDP where every state, the action has been chosen for you. There's no choice.

### Lesson 2: Sampling-based Approximations and Function Fitting
#### Notes from lesson
([video](https://www.youtube.com/watch?v=qO-HUo0LsO4))

Given the limitations of the previous lesson we'll react as follows:
Limitations are:
- Iteration over/storage for all states, actions and rewards require **small and discrete
state-action space** -> **sampling-based approximations**
- Update equations require access to the dynamical model (of the robot or whatever the agent is) -> **Q/V function fitting methods**.


Given Q-value iteration as:
$$
Q_{k+1}(s,a) \leftarrow  \sum_{s'} P(s' | s,a) \cdot (R(s,a,s') + \gamma \underset{a'}{max} Q_k (s',a') )
$$

This equation is pretty expensive from a computational perspective. Consider a robot, you'll need to know each one of the possible future states and compute the corresponding probability of reaching those states.

Rewrite it as an expectation:
$$
Q_{k+1}(s,a) \leftarrow  \mathbb{E}_{s' \approx P(s'|s,a)} [R(s,a,s') + \gamma \underset{a'}{max} Q_k (s',a') ]
$$

This results in an algorithm called "tabular Q-learning" whose algorithm follows:

##### Algorithm: Tabular Q-Learning

- Start with $Q_0 (s,a)$ for all $s$, $a$.
- Get initial state $s$
- For $k=1,2, ...$ till convergence:
  - Sample action $a$, get next state $s'$
  - If $s'$ is terminal:
    - $target = R(s,a,s')$
    - Sample new initial state $s'$
  - else:
    - $target = R(s,a,s') + \gamma \underset{a'}{max} Q_k (s',a')$
  - $Q_{k+1} (s,a) \leftarrow (1 - \alpha) \cdot Q_k (s,a) + \alpha[target]$
  - $s \leftarrow s'$


During the inital phases of learning, choosing greedily isn't optimal. If you only choose actions greedily you are restricting yourself to not explore alternative strategies.

Instead, what's used is $\epsilon-Greedy$.

Q-learning converges to optimal policy -- even if you're acting suboptimally!. This is called off-policy learning. Caveats:
- You have to explore enough
- You have to eventually make the learning rate small enough
- ... but not decrease it too fast, otherwise it will converge to wrong values.

The reinforcement learning performed by human race is a collective learning practice where we're not just looking at the experience of ourselves but also looking at the experience of other human beings.

Getting tabular Q-learning to work is unfeasible for most environments thereby "Approximate Q-Learning" is introduced. This typically gets represented through a parametrized Q-function: $Q_\theta(s,a)$ that can get represented with linear functions or with more complicated neural nets.

In some case, "Tabular Q-learning" is a special case of "Approximate Q-learning".

### Lesson 3: Deep Q-Networks
#### Notes from lesson
([video](https://www.youtube.com/watch?v=fevMOp5TDQs&t=3s))

The whole idea behind DQN is to make Q-learning look like it's supervised learning.

Two ideas to address the correlation between updates:
- Experience replay -> better data efficiency
- Use older set of weights to compute the targets (target network). This is good because if we fix this target weights then basically we have a fixed network that's producing our estimates and we have also those experience replay buffers, something like a dataset which we're sampling data from. So what we do with this is that we feed this Q values to something that almost looks like a fixed set of targets.

When one runs DQN in a new problem it might $10^6$ transitions to get some reasonable results.

##### Algorithm: Deep Q-Learning (DQN)
- Initialize replay memory $D$ to capacity $N$
- Initialize action-value function Q with random weights $\theta$
- Initialize target action-value function \hat{Q} with weights $\hat{\theta} = \theta$
- **For** $episode = 1,M$ **do**:
  - Initialize sequence $s_1 = {x_1}$ and preprocessed sequence $\phi_1 = \phi(s_1)$
  - **For** $t = 1,T$ **do**
    - With probability $\epsilon$ select a random action $a_t$
    - otherwise select $a_t = argmax_a Q(\phi(s_t),a;\theta)$
    - Execute action $a_t$ in emulator and observe reward $r$, and image $x_{t+1}$
    - Set $s_{t+1} = s, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$
    - Store transition ($\phi_t, a_t, r_t, \phi_{t+1}$) in $D$
    - Sample random minibatch of transition (($\phi_j, a_j, r_j, \phi_{j+1}$))
    - Set $$
    y_j = \left\{
	       \begin{array}{ll}
      		 r_j      & \mathrm{if\ } episode terminates at step j + 1\\
      		 r_j + \gamma max_{a'} \hat{Q}(\phi_{j+1},a';\hat{\theta}) & \mathrm{otherwise\ }
	       \end{array}
	     \right.
       $$
    - Perform a gradient descent on $(y_j - Q(\phi_j, a_j; \theta))^2$ with respect to the network parameters $\theta$
    - Every $C$ steps reset $\hat{Q} = Q$
  - **End For**
- **End For**

Preprocessed $\phi$ elements correspond to 4 frames of the game stacked together as an input to the network
representing the state (this worked for them).

Value-based methods tend to be more robust parameter-wise. Much more than policy gradient methods. People in DeepMind is running algorithms with barely no variations on the hyperparams.

Double DQN, an upgrade of DQN. It exploits the fact that you have two networks: the online network and the target network. The idea is to use your online network to select the best action but then you use the target network to get the value estimate. You separate the $argmax$ from selecting the value.

Dueling DQN, make two separate channels:
- a channel that output a single channel, the value
- and another channel that outputs one number per action, the advantage
Summing this up (to get the output estimate of the Q-value) it will work much better in practice.

You can go beyond $\epsilon-Greedy$ exploration, one good way of doing exploration with NNs is to add noise to the parameters of the NNs.
