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
while it doesn't depend on the action. The value function $V_\pi$ tells you how much reward you'll get.

Explanation at https://youtu.be/S_gwYj1Q-44?t=36m:
##### Algorithm: Vanilla Policy Gradient [[William, 1992](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)]
- Initialize policy (e.g. NNs) parameter $\theta$ and baseline $b$
- **for** $iteration=1,2,...$ **do**
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

In total: $[(80 \cdot 80)\cdot 200 + 200] + (200)\cdot 1 + 1 = 1282201 \approx 1.3 M $.

There's no way to learn from static frames so what they do is concatenate frames together or use the difference.

Interesting way of putting Supervised Learning and Reinforcement Learning:

| Supervised Learning | Reinforcement Learning |
| --------------------|------------------------|
| We try to maximize: $\sum_i log p(y_i| x_i)$ for images $x_i$ and labels $y_i$ | we have no lables so we sample $y_i \approx p( ? | x_i)$ |
| | once we collect a batch of rollouts, we maximize: $\sum_i A_i \cdot log p(y_i | x_i)$ |

where $A_i$ is called the advantage (-1,+1) depending on the result.

Discounting is heuristic to do a modulation of the blame for a bad (or good) reward. Typically represented as $\gamma$.

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
For $interation=1,2,...$ do
- Run policy for $T$ timesteps or $N$ trajectories
- Estimate advantage function at all timesteps:
$$
   \underset{\theta}{maximize} \sum_{n=1}^{N} \frac{\pi_\theta (a_n | s_n)}{\pi_{\theta_old}(a_n | s_n)} \hat(A)_n
$$
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
For $iteration=1,2,...$ do
- Run policy for $T$ timesteps or $N$ trajectories
- Estimate advantage function at all timesteps
- DO SGD on $L^{CLIP}(\theta)$ objective for some number of epochs
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
- Rescale observations with: $x' = clip((x- \mu)/\sigma, -10,10)$
- Rescale rewards but don't shift the mean as that affects agents will to live
- Standardize prediction targets *is more complicated*.

Generally important parameters:
- Watch out with $\gamma$, it can ignore rewards delayed by x timesteps
- Action frequency, make sure it's human solvable.

General RL diagnostics:
- Look at episode returns min/max and stdev along with mean.
- Look at episode lengths (*sometimes more informative that the reward*), sometimes provides additional information (e.g.: solving problem faster, losing game slower)

#### Policy gradient strategies

Entropy as a diagnostic:
- If your entropy is going down really fast it means the policy is becoming deterministic and it's not going to explore anything. Be careful also if it's not going down (totally random always). **How do you measure entropy?** For most policies you can compute the entropy analytically. For continuous gaussian policies you can compute the differencial entropies.
- Use entropy bonus

Baseline explained variance:
$$
explained_variance = \frac{1 - Var[empirical_return - predicted_value]}{Var[empirical_return]}
$$

Policy initialization is relevant. Specially in supervised learning. Zero or tiny final layer, to maximize entropy.

#### Q-Learning strategies
- optimize memory usage (replay buffer)
- learning rate,
- schedules are often useful ($\epsilon-Greedy$)
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
$$
Q(s,a) = E[R(s,a,s')] + \gamma E[Q(s',a')]
$$

$Z(s,a)$ will represent the distribution of the return of taking action $a$ in state $s$. The distribution of the Bellman equation can be expressed as:
$$
Z(s,a) = R(s,a,S') + \gamma Z(S',A')
$$

(*capital letters are representing random variables*)

C51 is a specific instantiation of using this previous idea of distributions:
- C51 outputs a 51-way softmax for each Q-value.
- Bin the return into 51 equally-sized bins between -10 and 10
- The Q-value update becomes categorial (instead of the usual regression)

Pseudocode: Categorical Algorithm
- input A transition $x_t$, $a_t$, $r_t$, $x_{t+1}$, $\gamma_t \in [0,1]$
  - $Q(x_{t+1},a) := \sum_i z_i p_i (x_{t+1},a)$
  - $a^* \rarrow argmax_a Q(x_{t+1},a)$
  - $m_i = 0, i \in 0,...,N-1$
  - $for j \in 0,...,N -1$ do
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
