# A review of different AI techniques for RL

This tutorial will review the State Of The Art (SOTA) of RL using the [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures). Techniques
will get benchmarked using OpenAI gym-based environments.

### Lesson 1: Deep RL Bootcamp Lecture 1: Motivation + Overview + Exact Solution Methods
([video](https://www.youtube.com/watch?v=qaMdN6LS9rA))

$Q^*(s,a) = $ expected utility/reward/value starting in $s$, taking action $a$ and (thereafter)
acting optimally.

The Bellman Equation:
$$
Q^*(s,a) = \sum_{s'} P(s' | s,a) \cdot (R(s,a,s') + \gamma \underset{below}{above} )
$$
