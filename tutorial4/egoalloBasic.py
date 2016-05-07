import numpy as np
import sys
import random

import cellular
import qlearn

startCell = None

class Cell(cellular.Cell):
    def __init__(self):
        self.cliff = False
        self.goal = False
        self.wall = False

    def colour(self):
        if self.cliff:
            return 'red'
        if self.goal:
            return 'green'
        if self.wall:
            return 'black'
        else:
            return 'white'

    def load(self, data):
        global startCell
        if data == 'S':
            startCell = self
        if data == '.':
            self.wall = True
        if data == 'X':
            self.cliff = True
        if data == 'G':
            self.goal = True


class Agent(cellular.Agent):
    def __init__(self, allo_weight=.5, ego_weight=.5, weight_learning=True):

        self.actions = range(directions)
        self.epsilon = .1
        self.eta = 5e-6
        self.lastAction = None
        self.lastaction_index = None
        self.hitWall = False
        self.score = 0
        self.intentional_deaths = 0
        self.unintentional_deaths = 0
        self.intentional = True

        self.allo_weight = allo_weight
        self.alloAI = qlearn.QLearn(
            actions=self.actions, epsilon=0.05, alpha=0.1, gamma=.9)
        self.ego_weight = ego_weight
        self.egoAI = qlearn.QLearn(
            actions=self.actions, epsilon=0.05, alpha=0.1, gamma=.0)
        self.weight_learning = weight_learning

    def colour(self):
        return 'blue'

    def update(self):

        reward = self.calcReward()
        alloState = self.calcAlloState()
        egoState = self.calcEgoState()

        if self.lastAction is not None:

            dq_allo = self.alloAI.learn(self.lastAlloState, self.lastAction, 
                    reward, alloState)

            dq_ego = self.egoAI.learn(self.lastEgoState, self.lastAction, 
                    reward, egoState)

            if self.weight_learning == True:
                # update weightings 
                self.allo_weight += self.eta * reward * dq_allo * (self.ego_q[self.lastaction_index])
                self.allo_weight = max(0, self.allo_weight)
                self.ego_weight += self.eta * reward * dq_ego * (self.allo_q[self.lastaction_index])
                self.ego_weight = max(0, self.ego_weight)

                # normalize weightings
                norm = abs(self.allo_weight) + abs(self.ego_weight)
                self.allo_weight = self.allo_weight / norm
                self.ego_weight = self.ego_weight / norm

        if self.cell.goal == True:
            self.score += 1
            self.restart()

        elif self.cell.cliff == True:
            if self.intentional:
                self.intentional_deaths += 1
            else:
                self.unintentional_deaths += 1

            self.restart()

        else: 

            # calculate action values based on scaled 
            # summation of ego and allo
            self.allo_q = [self.alloAI.getQ(alloState, a) for a in self.actions]
            self.ego_q = [self.egoAI.getQ(egoState, a) for a in self.actions]
            q = [self.allo_weight * allo_val + self.ego_weight * ego_val
                    for allo_val, ego_val in zip(self.allo_q, self.ego_q)]

            self.intentional = True
            if random.random() < self.epsilon:
                self.lastaction_index = np.random.randint(len(self.actions))
                self.intentional = False

            else:
                maxQ = max(q)
                count = q.count(maxQ)
                if count > 1:
                    best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                    self.lastaction_index = random.choice(best)
                else:
                    self.lastaction_index = q.index(maxQ)

            action = self.actions[self.lastaction_index]
            
            self.lastAlloState = alloState 
            self.lastEgoState = egoState
            self.lastAction = action

            self.hitWall = not self.goInDirection(action)


    def calcAlloState(self):
        return self.cell.x, self.cell.y

    def calcEgoState(self):
        return tuple([self.cellvalue(c) for c in self.cell.neighbours])

    def calcReward(self):
        if self.cell.cliff == True:
            return cliffReward
        elif self.cell.goal == True:
            return goalReward
        elif self.hitWall == True:
            return hitWallReward
        else:
            return normalReward

    def cellvalue(self, cell):
        return (3 if cell.goal else
                2 if cell.wall else
                1 if cell.cliff else 
                0)

    def restart(self):
        self.cell = startCell
        self.lastAction = None
        self.lastEgoState = None
        self.lastaction_index = None

cliffReward = -10
goalReward = 500
hitWallReward = -5
normalReward = -1

directions = 4

time_limit = 50e4
sample_every = 10e3
average_across = 10
test_args = [
             # only allocentric
             {'allo_weight':1.0, 
              'ego_weight':0.0, 
              'weight_learning':False}, 
             # only egocentric 
             {'allo_weight':0.0, 
              'ego_weight':1.0, 
              'weight_learning':False},
             # .5 allo + .5 ego
             {'allo_weight':0.5, 
              'ego_weight':0.5, 
              'weight_learning':False},
             # dynamic weighting
             {'allo_weight':0.5, 
              'ego_weight':0.5, 
              'weight_learning':True}]


import scipy.stats
def mean_and_confint(data):
    '''calculate and return mean and confidence intervals of data'''
    n, min_max, mean, var, skew, kurt = scipy.stats.describe(data)
    std = np.sqrt(var)
    
    R = scipy.stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(len(data)))
    return mean, R[0], R[1]



win_stats = np.zeros((len(test_args), time_limit / sample_every, 3))
suicide_stats = np.zeros((len(test_args), time_limit / sample_every, 3))
accident_stats = np.zeros((len(test_args), time_limit / sample_every, 3))



for ii, args in enumerate(test_args): 
    print '\n----------------------------------'
    print 'testing argument set %i'%ii
    print args
    print '----------------------------------'

    stats_avg = np.zeros((average_across, time_limit / sample_every, 3))
    for jj in range(average_across):
        print 'trial %i'%jj
        
        index = 0

        world = cellular.World(Cell, directions=directions, filename='worlds/barrier3.txt')

        if startCell is None:
            print "You must indicate where the agent starts by putting a 'S' in the map file"
            sys.exit()
        agent = Agent(**args)
        world.addAgent(agent, cell=startCell)

        pretraining = 0
        for i in range(pretraining):
            if i % 1000 == 0:
                print i, agent.score
                agent.score = 0
            world.update()

        ### display
        world.display.activate(size=30)
        world.display.delay = 1

        while world.age < time_limit + 1:
            world.update()

            if world.age % sample_every == 0:
                print "{:d}, W: {:d}, L: {:d}, A: {:d}"\
                    .format(world.age, agent.score, agent.intentional_deaths, agent.unintentional_deaths)
                stats_avg[jj, index] = np.array([agent.score, agent.intentional_deaths, agent.unintentional_deaths])
                print "allo: %.3f, ego: %.3f, epsilon: %.3f"%(agent.allo_weight, agent.ego_weight, agent.epsilon)
                agent.score = 0
                agent.intentional_deaths = 0
                agent.unintentional_deaths = 0
                index += 1
            if world.age % 100000 == 0: 
                agent.epsilon /= 2

    # stats_total[ii] = np.sum(stats_avg, axis=0) / average_across
    for kk in range(int(time_limit / sample_every)):
        win_stats[ii, kk] = mean_and_confint(stats_avg[:, kk, 0])
        suicide_stats[ii, kk] = mean_and_confint(stats_avg[:, kk, 1])
        accident_stats[ii, kk] = mean_and_confint(stats_avg[:, kk, 2])

import matplotlib.pyplot as plt
x = range(0, int(time_limit), int(sample_every))

for ii in range(len(test_args)):
    plt.figure()
    
    # plot successes means and confidence intervals
    plt.subplot(311)
    plt.fill_between(x, win_stats[ii, :, 1], win_stats[ii, :, 2], alpha=.5)
    plt.plot(x, win_stats[ii, :, 0], 'k', lw=3)
    plt.ylabel('wins')

    plt.title(test_args[ii])

    # plot successes means and confidence intervals
    plt.subplot(312)
    plt.fill_between(x, suicide_stats[ii, :, 1], suicide_stats[ii, :, 2], alpha=.5)
    plt.plot(x, suicide_stats[ii, :, 0], 'k',  lw=3)
    plt.ylabel('suicides')

    # plot successes means and confidence intervals
    plt.subplot(313)
    plt.fill_between(x, accident_stats[ii, :, 1], accident_stats[ii, :, 2], alpha=.5)
    plt.plot(x, accident_stats[ii, :, 0], 'k', lw=3)
    plt.ylabel('accidents')
    plt.xlabel('simulation steps')

plt.tight_layout()
plt.show()
