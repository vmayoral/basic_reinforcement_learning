# A review of different AI techniques for RL

This tutorial will review the State Of The Art (SOTA) of RL using the [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures). Techniques
will get benchmarked using OpenAI gym-based environments.

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

##### Algorithm

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
