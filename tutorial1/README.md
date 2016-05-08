Basic Reinforcement Learning Tutorial 1: Q-learning
===================================================

This tutorial covers the basics of Q-learning using the cat-mouse-cheese example. 

## Table of Contents
1. [The *World* and *Cat* player implementations](#world)
2. [The *Mouse* player](#mouse)
3. [Results](#results)

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

<div id='results'/>
### Results
Below we present a *mouse player* after **15 000 generations** of reinforcement learning:

![](../img/rl_qlearning_1.gif)

and now we present the same player after **150 000 generations** of reinforcement learning:

![](../img/rl_qlearning_2.gif)
