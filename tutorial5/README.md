Tutorial 5: Deep Q-learning
========

Up until this tutorial Q-learning algorithm has been storing state-action pairs in a table, a dictionary or a similar kind of data structure. The fact is that there're many scenarios where tables don't scale nicely. Let's take Pacman. If we implement it as a graphics-based game, the state would be the raw pixel data. In a tabular method, if the pixel data changes by just a single pixel, we have to store that as a completely separate entry in the table. Obviously that's silly and wasteful. What we need is **some way to generalize and pattern match between states**. We need our algorithm to say "the value of these kind of states is X" rather than "the value of this exact, super specific state is X."

That's where neural networks come in. Or any other type of function approximator, even a simple linear model. We can use a neural network, instead of a lookup table, as our  Q(s,a)Q(s,a)  function. Just like before, it will accept a state and an action and spit out the value of that state-action.

### Comparing different techniques in `CartPole`


| Algorithm | `epochs:` 100 | `epochs:` 500  | `epochs:` 1000  |
|-----------|----------------|----------------|-----------------|
| Q-learning| 104.87 (86.37) | 181.22 (145.78) | 191.35 (141.31) |
| DQN	| | |

*Each cell represents the best 100 scores for the number of epochs and in parenthesis the average score over all the epochs*


- Intro to Theano
- Intro MNIST in Theano
- Intro to MNIST in Keras
- TODO: DQN with Keras in OpenAI gym

### References:
- http://deeplearning.net/tutorial/mlp.html#mlp
- http://outlace.com/Reinforcement-Learning-Part-3/
- http://keras.io/
- https://github.com/sherjilozair/dqn

