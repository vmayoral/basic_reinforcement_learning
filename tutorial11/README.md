# A review of different AI techniques for RL

This tutorial will review the State Of The Art (SOTA) of RL using the [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures). Techniques
will get benchmarked using OpenAI gym-based environments.

### Lesson 1: Deep RL Bootcamp Lecture 1: Motivation + Overview + Exact Solution Methods
([video](https://www.youtube.com/watch?v=qaMdN6LS9rA))

<img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/83298bfbf00ad918af5dfff006b94b06.svg?invert_in_darkmode" align=middle width=74.390085pt height=24.65759999999998pt/> expected utility/reward/value starting in <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/6f9bad7347b91ceebebd3ad7e6f6f2d1.svg?invert_in_darkmode" align=middle width=7.705549500000004pt height=14.155350000000013pt/>, taking action <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.689230000000004pt height=14.155350000000013pt/> and (thereafter)
acting optimally.

The Bellman Equation:
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/fa715d6477ee55df4ae3e19ebecd524a.svg?invert_in_darkmode" align=middle width=334.45995pt height=36.895155pt/></p>
