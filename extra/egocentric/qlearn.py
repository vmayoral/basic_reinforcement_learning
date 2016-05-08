import random


class QLearn:
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.q = {}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learn(self, state, action, reward, state2):

        if self.q.has_key((state, action)) == False:
            self.q[(state, action)] = reward

        else:

            maxqnew = max([self.getQ(state2, a) for a in self.actions])
            value = reward + self.gamma * maxqnew
            self.q[(state, action)] += self.alpha * (value - self.q[(state, action)])

    def chooseAction(self, state):

        intentional = True
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            intentional = False

        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action, intentional

