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

<img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/83298bfbf00ad918af5dfff006b94b06.svg?invert_in_darkmode" align=middle width=74.390085pt height=24.65759999999998pt/> expected utility/reward/value starting in <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/6f9bad7347b91ceebebd3ad7e6f6f2d1.svg?invert_in_darkmode" align=middle width=7.705549500000004pt height=14.155350000000013pt/>, taking action <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.689230000000004pt height=14.155350000000013pt/> and (thereafter)
acting optimally.


This Q-value function satisfies the Bellman Equation:
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/aafd0d9eec7fffc4862e91ca58e5a81f.svg?invert_in_darkmode" align=middle width=390.70185pt height=36.895155pt/></p>

For solving <img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/e6f426deb3194d674706c8c9ea55188c.svg?invert_in_darkmode" align=middle width=19.730700000000002pt height=22.638659999999973pt/>, an interative method named "Q-value iteration" is proposed.

<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/7c7f118952a68495683f525472a135b0.svg?invert_in_darkmode" align=middle width=408.40635pt height=36.895155pt/></p>

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
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/7c7f118952a68495683f525472a135b0.svg?invert_in_darkmode" align=middle width=408.40635pt height=36.895155pt/></p>

This equation is pretty expensive from a computational perspective. Consider a robot, you'll need to know each one of the possible future states and compute the corresponding probability of reaching those states.

Rewrite it as an expectation:
<p align="center"><img src="https://rawgit.com/vmayoral/basic_reinforcement_learning/master//tutorial11/tex/55b27d56cb02d27fe9b724ac75c6acb9.svg?invert_in_darkmode" align=middle width=385.10339999999997pt height=23.043074999999998pt/></p>
