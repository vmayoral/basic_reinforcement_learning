Basic Reinforcement Learning Tutorial 1: Q-learning
===================================================

This tutorial covers the basics of Q-learning using the cat-mouse-cheese example. 

## Table of Contents
- [The *World* and *Cat* player implementations](#world)
- [The *Mouse* player](#mouse)
  - [Q-learning implementation](#q-learning)
- [Results](#results)
- [Reproduce it yourself](#reproduce)

<div id='world'/>
### The *World* and *Cat* player implementations

The implementations of the discrete 2D world (including agents, cells and other abstractions) as well as the cat and mouse players is performed in the `cellular.py` file. The world is generated from a `.txt`file. In particular, I'm using the `worlds/waco.txt`:

```
(waco world)

XXXXXXXXXXXXXX
X            X
X XXX X   XX X
X  X  XX XXX X
X XX      X  X
X    X  X    X
X X XXX X XXXX
X X  X  X    X
X XX   XXX  XX
X    X       X
XXXXXXXXXXXXXX

```

The *Cat* player class inherit from `cellular.Agent` and its implementation is set to follow the *Mouse* player:

```python

    def goTowards(self, target):
        if self.cell == target:
            return
        best = None
        for n in self.cell.neighbours:
            if n == target:
                best = target
                break
            dist = (n.x - target.x) ** 2 + (n.y - target.y) ** 2
            if best is None or bestDist > dist:
                best = n
                bestDist = dist
        if best is not None:
            if getattr(best, 'wall', False):
                return
            self.cell = best

```
The *Cat* player calculates the quadratic distance (`bestDist`)  among its neighbours and moves itself (`self.cell = best`) to that cell.

```python
class Cat(cellular.Agent):
    cell = None
    score = 0
    colour = 'orange'

    def update(self):
        cell = self.cell
        if cell != mouse.cell:
            self.goTowards(mouse.cell)
            while cell == self.cell:
                self.goInDirection(random.randrange(directions))

```
Overall, the *Cat* pursues the *Mouse* through the `goTowards` method by calculating the quadratic distance. Whenever it bumps to the wall, it takes a random action.

<div id='mouse'/>
### The *Mouse* player

The *Mouse* player contains the following attributes:
```python
class Mouse(cellular.Agent):
    colour = 'gray'

    def __init__(self):
        self.ai = None
        self.ai = qlearn.QLearn(actions=range(directions),
                                alpha=0.1, gamma=0.9, epsilon=0.1)
        self.eaten = 0
        self.fed = 0
        self.lastState = None
        self.lastAction = None

```
The `eaten` and `fed` attributes store the performance of the player while the `lastState` and `lastAction` ones help the *Mouse* player store information about its previous states (later used for learning).

The `ai` attribute stores the Q-learning implementation which is initialized with the following parameters:
- directions: There're different possible directions/actions implemented at the `getPointInDirection` method of the *World* class:

```python
    def getPointInDirection(self, x, y, dir):
        if self.directions == 8:
            dx, dy = [(0, -1), (1, -1), (
                1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)][dir]
        elif self.directions == 4:
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][dir]
        elif self.directions == 6:
            if y % 2 == 0:
                dx, dy = [(1, 0), (0, 1), (-1, 1), (-1, 0),
                          (-1, -1), (0, -1)][dir]
            else:
                dx, dy = [(1, 0), (1, 1), (0, 1), (-1, 0),
                          (0, -1), (1, -1)][dir]


```
In general, this implementaiton will be used in **8 directions**, thererby `(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]`.
- alpha: which is the q-learning discount constant, set to *0.1*.
- gamma: 
- epsilon: an exploration constant to randomize decisions, set to *0.1*.


<div id='q-learning'/>
#### Q-learning implementation

<div id='results'/>
### Results
Below we present a *mouse player* after **15 000 generations** of reinforcement learning:

![](../img/rl_qlearning_1.gif)

and now we present the same player after **150 000 generations** of reinforcement learning:

![](../img/rl_qlearning_2.gif)

<div id='reproduce'/>
### Reproduce it yourself

```bash
git clone https://github.com/vmayoral/basic_reinforcement_learning
cd basic_reinforcement_learning
python tutorial1/egoMouseLook.py
```