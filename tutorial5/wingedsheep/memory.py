import numpy as np
import random

class Memory:
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