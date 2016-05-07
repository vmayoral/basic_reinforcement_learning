import cellular
import qlearn
import time
import random
import shelve

directions = 8

def pickRandomLocation():
    while 1:
        x = random.randrange(world.width)
        y = random.randrange(world.height)
        cell = world.getCell(x, y)
        if not (cell.wall or len(cell.agents) > 0):
            return cell


class Cell(cellular.Cell):
    wall = False

    def colour(self):
        if self.wall:
            return 'black'
        else:
            return 'white'

    def load(self, data):
        if data == 'X':
            self.wall = True
        else:
            self.wall = False


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


class Cheese(cellular.Agent):
    colour = 'yellow'

    def update(self):
        pass


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

    def update(self):
        state = self.calcState()
        reward = -1

        if self.cell == cat.cell:
            self.eaten += 1
            reward = -100
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, reward, state)
            self.lastState = None

            self.cell = pickRandomLocation()
            return

        if self.cell == cheese.cell:
            self.fed += 1
            reward = 50
            cheese.cell = pickRandomLocation()

        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, reward, state)

        state = self.calcState()
        action = self.ai.chooseAction(state)
        self.lastState = state
        self.lastAction = action

        self.goInDirection(action)

    def calcState(self):
        if cat.cell is not None:
            return self.cell.x, self.cell.y, cat.cell.x, cat.cell.y, cheese.cell.x, cheese.cell.y
        else:
            return self.cell.x, self.cell.y, cheese.cell.x, cheese.cell.y


mouse = Mouse()
cat = Cat()
cheese = Cheese()

world = cellular.World(Cell, directions=directions, filename='barrier2.txt')
world.age = 0

world.addAgent(cheese, cell=pickRandomLocation())
world.addAgent(cat)
world.addAgent(mouse)

epsilonx = (0,100000)
epsilony = (0.1,0)
epsilonm = (epsilony[1] - epsilony[0]) / (epsilonx[1] - epsilonx[0])

endAge = world.age + 150000
while world.age < endAge:
    world.update()

    if world.age % 100 == 0:
        mouse.ai.epsilon = (epsilony[0] if world.age < epsilonx[0] else
                            epsilony[1] if world.age > epsilonx[1] else
                            epsilonm*(world.age - epsilonx[0]) + epsilony[0])

    if world.age % 10000 == 0:
        print "{:d}, e: {:0.2f}, W: {:d}, L: {:d}"\
            .format(world.age, mouse.ai.epsilon, mouse.fed, mouse.eaten)
        mouse.eaten = 0
        mouse.fed = 0

mouse.ai.epsilon = 0.0    # change this to 0 to turn off exploration after learning

world.display.activate(size=30)
world.display.delay = 1
while 1:
    world.update()
