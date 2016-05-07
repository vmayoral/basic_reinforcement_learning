import sys

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
    def __init__(self):
        self.actions = range(directions)
        self.egoAI = qlearn.QLearn(
            actions=self.actions, epsilon=0.05, alpha=0.1, gamma=.05)
        self.lastAction = None
        self.hitWall = False
        self.score = 0
        self.intentional_deaths = 0
        self.unintentional_deaths = 0
        self.intentional = True

    def colour(self):
        return 'blue'

    def update(self):

        reward = self.calcReward()
        egoState = self.calcEgoState()

        if self.lastAction is not None:
            self.egoAI.learn(self.lastEgoState, self.lastAction, 
                    reward, egoState)

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

            action, self.intentional = self.egoAI.chooseAction(egoState)

            self.lastEgoState = egoState
            self.lastAction = action

            self.hitWall = not self.goInDirection(action)

            # # get new state and print q-values
            # egoState = self.calcEgoState()
            # reward = self.calcReward()
            # q = [self.egoAI.getQ(egoState, a) for a in self.egoAI.actions]
            # print "          %.3f     "%q[0]
            # print "  %.3f  %.3f  %.3f   "%(q[3], reward, q[1])
            # print "          %.3f     "%q[2]
            # print ""

    def calcEgoState(self):
        return tuple([self.cellvalue(c) for c in self.cell.neighbours])

    def calcReward(self):
        if self.cell.cliff == True:
            return cliffReward
        elif self.cell.goal == True:
            return goalReward
        elif self.hitWall == True:
            # print 'hitwall reward'
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

cliffReward = -10
goalReward = 500
hitWallReward = -5
normalReward = -1

directions = 4
world = cellular.World(Cell, directions=directions, filename='cliffs.txt')

if startCell is None:
    print "You must indicate where the agent starts by putting a 'S' in the map file"
    sys.exit()
agent = Agent()
world.addAgent(agent, cell=startCell)

pretraining = 0
for i in range(pretraining):
    if i % 1000 == 0:
        print i, agent.score
        agent.score = 0
    world.update()

### display
print cellular.Display
world.display.activate(size=30)
world.display.delay = 1
while 1:
    world.update()

    if world.age % 10000 == 0:
        print "{:d}, W: {:d}, L: {:d}, A: {:d}"\
            .format(world.age, agent.score, agent.intentional_deaths, agent.unintentional_deaths)
        agent.score = 0
        agent.intentional_deaths = 0
        agent.unintentional_deaths = 0
