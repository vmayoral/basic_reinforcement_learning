# A review of different AI techniques for RL

This tutorial will review the State Of The Art (SOTA) of RL using the [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures). Techniques
will get benchmarked using OpenAI gym-based environments.

### Lesson 1: Deep RL Bootcamp Lecture 1: Motivation + Overview + Exact Solution Methods
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

The Bellman Equation for Q-values:
$$
Q^*(s,a) \leftarrow  \sum_{s'} P(s' | s,a) \cdot (R(s,a,s') + \gamma \underset{max}{a'} Q^* (s',a') )
$$

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
([video](https://www.youtube.com/watch?v=qO-HUo0LsO4))
