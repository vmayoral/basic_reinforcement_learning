'''
Evolutionary deep artificial neural network approach to the CartPole problem
using OpenAI's gym environment.

As part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning

Code based on @wingedsheep's Simple NeuroEvolution Cartpole work at 
https://gist.github.com/wingedsheep/426e847e193d79c67e052a856d495338

        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''

# import the neural net stuff
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.models import model_from_json

# import other stuff
import random
import numpy as np
import gym

# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
# import theano

class Memory:
    """
    This class provides an abstraction to store the [s, a, r, a'] elements of each iteration.
    Instead of using tuples (as other implementations do), the information is stored in lists 
    that get returned as another list of dictionaries with each key corresponding to either 
    "state", "action", "reward", "nextState" or "isFinal".
    """
    def __init__(self, size):
        self.size = size
        self.currentPosition = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.newStates = []
        self.finals = []

    def getMiniBatch(self, size) :
        indices = random.sample(np.arange(len(self.states)), min(size,len(self.states)) )
        miniBatch = []
        for index in indices:
            miniBatch.append({'state': self.states[index],'action': self.actions[index], 'reward': self.rewards[index], 'newState': self.newStates[index], 'isFinal': self.finals[index]})
        return miniBatch

    def getCurrentSize(self) :
        return len(self.states)

    def getMemory(self, index): 
        return {'state': self.states[index],'action': self.actions[index], 'reward': self.rewards[index], 'newState': self.newStates[index], 'isFinal': self.finals[index]}

    def addMemory(self, state, action, reward, newState, isFinal) :
        if (self.currentPosition >= self.size - 1) :
            self.currentPosition = 0
        if (len(self.states) > self.size) :
            self.states[self.currentPosition] = state
            self.actions[self.currentPosition] = action
            self.rewards[self.currentPosition] = reward
            self.newStates[self.currentPosition] = newState
            self.finals[self.currentPosition] = isFinal
        else :
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.newStates.append(newState)
            self.finals.append(isFinal)
        
        self.currentPosition += 1

class Agent:
    """
    Agent class for the NeuroEvolution algorithm. 
    Each agent represents an artificial neural network.
    """
    def __init__(self, network=None, fitness=None):
        self.network = network
        self.fitness = fitness

    def setNetwork(self, network):
        self.network = network

    def setFitness(self, fitness):
        self.fitness = fitness

    def __repr__(self):
        return 'Agent {0}'.format(self.fitness)

class Evolution:
    """
    Evolution class

    Parameters:
        - inputs: number of inputs
        - outputs: number of outputs
        - nr_rounds_per_epoch: number of rounds used to evaluate the fitness 
            of each agent (5 by default)
        - env: environment (OpenAI gym's env instance)
        - steps: number of steps to simulate (200 in the case of OpenAI's CartPole)
        - epochs: number of epochs for the evolution (could stop before this number of epochs)
        - scoreTarget (1e9): Find a best average fitness above this value (1e9 by default)
    """
    def __init__(self, inputs, outputs, nr_rounds_per_epoch, env, steps, epochs, scoreTarget = 1e9):
        self.input_size = inputs
        self.output_size = outputs
        self.nr_rounds_per_epoch = nr_rounds_per_epoch
        self.env = env
        self.steps = steps
        self.epochs = epochs
        self.scoreTarget = scoreTarget
   
    def initAgents(self, nr_agents, hiddenLayers):
        """
        Initializes nr_agent agents to an ANN, each with hiddenLayers structure
        """
        self.nr_agents = nr_agents
        self.agents = [None] * nr_agents
        self.hiddenLayers = hiddenLayers

        for i in xrange(self.nr_agents):
            agent = Agent()
            model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu")
            agent.setNetwork(model)
            self.agents[i] = agent

    def createRegularizedModel(self, inputs, outputs, hiddenLayers, activationType):
        bias = False
        dropout = 0.1
        regularizationFactor = 0
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        else :
            if regularizationFactor > 0:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', W_regularizer=l2(regularizationFactor),  bias=bias))
            else:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', bias=bias))

            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
            
            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                if regularizationFactor > 0:
                    model.add(Dense(layerSize, init='lecun_uniform', W_regularizer=l2(regularizationFactor), bias=bias))
                else:
                    model.add(Dense(layerSize, init='lecun_uniform', bias=bias))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
                if dropout > 0:
                    model.add(Dropout(dropout))
            model.add(Dense(self.output_size, init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def createModel(self, inputs, outputs, hiddenLayers, activationType):
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
            model.add(Activation("linear"))
        else:
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else:
                model.add(Activation(activationType))
            
            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, init='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else:
                    model.add(Activation(activationType))
            model.add(Dense(self.output_size, init='lecun_uniform'))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=1, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        # model.summary()
        return model

    def backupNetwork(self, model, backup):
        weights = model.get_weights()
        backup.set_weights(weights)

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getValues(self, state, agent):
        predicted = agent.network.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getMaxValue(self, values):
        return np.max(values)

    def getMaxIndex(self, values):
        return np.argmax(values)

    # select the action with the highest value
    def selectAction(self, qValues):
        action = self.getMaxIndex(qValues)
        return action

    # could be useful in stochastic environments
    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def run_simulation(self, env, agent, steps, render = False):
        """
        Simply run the simulation and return the total reward
        """
        observation = env.reset()
        totalReward = 0
        for t in xrange(steps):
            if (render):
                env.render()
            values = self.getValues(observation, agent)
            action = self.selectAction(values)
            newObservation, reward, done, info = env.step(action)
            totalReward += reward
            observation = newObservation
            if done:
                break
        return totalReward

    def unroll(self, model):
        unrolled = []
        i = 0
        for layer in model.layers:
            if i % 2 == 0:
                weights = layer.get_weights()[0]
                for weight in weights:
                    for value in weight:
                        unrolled.append(value)
            i += 1
        self.roll(model, unrolled)
        return unrolled

    def roll(self, model, unrolled):
        new_weights = model.get_weights()
        for layer in model.layers:
            if i % 2 == 0:
                weights = layer.get_weights()[0]
                for weight in weights:
                    for value in weight:
                        unrolled.append(value)
            i += 1
        print new_weights

    def createWeightLayerList(self):
        weightLayerList = []
        for i in xrange(len(self.hiddenLayers)):
            weightLayerList.append(i * 2)
        return weightLayerList

    def crossover(self, agent1, agent2):
        new_weights = agent1.network.get_weights()
        agent2_weights = agent2.network.get_weights()
        for i in self.createWeightLayerList():
            layer2 = agent2_weights[i]
            for j in xrange(len(layer2)):
                rand = random.random()
                if (rand < 0.5):
                    new_weights[i][j] = layer2[j]
        child = Agent()
        newNetwork = self.createModel(self.input_size, self.output_size, self.hiddenLayers, "relu")
        newNetwork.set_weights(new_weights)
        child.setNetwork(newNetwork)
        return child

    def mutate(self, agent1):
        new_weights = agent1.network.get_weights()
        for i in self.createWeightLayerList():
            layer = new_weights[i]
            for j in xrange(len(layer)):
                neuronConnectionGroup = layer[j]
                for k in xrange(len(neuronConnectionGroup)):
                    weight = neuronConnectionGroup[k]
                    rand = random.random()
                    if (rand < 0.1):
                        rand2 = (random.random() - 0.5) * 0.1
                        new_weights[i][j][k] = neuronConnectionGroup[k] + rand2
        agent1.network.set_weights(new_weights)

    def selectBest(self):
        self.agents.sort(key=lambda x: x.fitness, reverse=True)
        selectionNr = int(self.nr_agents / 2)
        selectedAgents = self.agents[:selectionNr]
        return selectedAgents

    def createNewPopulation(self, bestAgents):
        print "create new pop"
        newPopulation = bestAgents
        while len(newPopulation) < self.nr_agents:
            rand = random.random()
            if rand < 0:
                parents = random.sample(bestAgents, 2) 
                child = self.crossover(parents[0], parents[1])
            elif rand < 0.67:
                parent = random.sample(bestAgents, 1)[0]
                child = Agent()
                newNetwork = self.createModel(self.input_size, self.output_size, self.hiddenLayers, "relu")
                newNetwork.set_weights(parent.network.get_weights())
                child.setNetwork(newNetwork)
                self.mutate(child)
            else:
                parents = random.sample(bestAgents, 2) 
                child = self.crossover(parents[0], parents[1])
                self.mutate(child)
            newPopulation.append(child)
        self.agents = newPopulation

    def tryAgent(self, agent, nr_episodes):
        total = 0
        for i in xrange(nr_episodes):
            total += self.run_simulation(self.env, agent, self.steps)
        return total / nr_episodes

    def evolve(self):
        """
        This function is in charge of implementing
        the evolutionary algorithm. 

        To do so it keeps (and evolves) the best half of the agents
        that performed best on every generation. The algorithm
        finishes when the best agent (computed over 100 episodes), obtains
        the maximum score (defined by self.scoreTarget).
        """
        for e in xrange(self.epochs):
            self.updateFitnessValuesForEpoch()
            averageFitness = self.calculateAverageFitness()
            print "Epoch",e,"average fitness: ",averageFitness
            bestAgents = self.selectBest()
            bestAgentAverage = self.tryAgent(bestAgents[0] , 100)
            if bestAgentAverage >= self.scoreTarget:
                print("bestAgentAverage: "+str(bestAgentAverage)+" >= self.scoreTarget "+str(self.scoreTarget))
                self.run_simulation(self.env, bestAgents[0] , self.steps*5, render = True)
                print("Finishing")
                break
            else:
                print "Best agent average: ",bestAgentAverage
            bestAgents[0].setFitness(bestAgentAverage)
            # self.run_simulation(self.env, bestAgents[0] , self.steps, render = True)
            self.createNewPopulation(bestAgents)

    def calculateAverageFitness(self):
        """
        Simply iterate over th agents and return the average fitness va
        """
        total = 0
        count = 0
        for index, agent in enumerate(self.agents):
            total += agent.fitness
            count += 1
        return total / count

    def updateFitnessValuesForEpoch(self):
        """
        run a number of simulations (nr_rounds_per_epoch) for each agent and determine their fitness (average score)
        by simply running the simulation.

        No learning happens in this step.
        """        
        agentScores = [0] * self.nr_agents
        for r in xrange(self.nr_rounds_per_epoch):
            for a in xrange(self.nr_agents):
                agent = self.agents[a]
                score = self.run_simulation(self.env, agent , self.steps)
                agentScores[a] += score

        for index, value in enumerate(agentScores):
            value /= self.nr_rounds_per_epoch
            self.agents[index].setFitness(value)


env = gym.make('CartPole-v0')

epochs = 100
steps = 200
scoreTarget = 200

# inputs, outputs, nr_rounds to determine fitness, env, max steps per episode, nr epochs, score target
evolution = Evolution(len(env.observation_space.high), env.action_space.n, 5, env, steps, epochs, scoreTarget)

# number of agents and hidden layer sizes in array
evolution.initAgents(100, [30, 30])
evolution.evolve()

# env.monitor.close()            