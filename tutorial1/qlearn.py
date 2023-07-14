import math
import random


class QLearn:
    """Q-Learning class. Implements the Q-Learning algorithm."""

    def __init__(self,
                 actions,
                 epsilon=0.1,
                 alpha=0.2,
                 gamma=0.9):
        """Initialize an empty dictionary for Q-Values."""
        # Q-Values are stored in a dictionary, with the state-action
        self.q = {}

        # Epsilon is the exploration factor. A higher epsilon
        # encourages more exploration, risking more but potentially
        # gaining more too.
        self.epsilon = epsilon

        # Alpha is the learning rate. If Alpha is high, then the
        # learning is faster but may not converge. If Alpha is low,
        # the learning is slower but convergence may be more stable.
        self.alpha = alpha

        # Gamma is the discount factor.
        # It prioritizes present rewards over future ones.
        self.gamma = gamma

        # Actions available in the environment
        self.actions = actions

    def getQ(self, state, action):
        """Get Q value for a state-action pair.

        If the state-action pair is not found in the dictionary,
            return 0.0 if not found in our dictionary
        """
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        """Updates the Q-value for a state-action pair.

        The core Q-Learning update rule.
            Q(s, a) += alpha * (reward(s,a) + max(Q(s')) - Q(s,a))

        This function updates the Q-value for a state-action pair
        based on the reward and maximum estimated future reward.
        """
        oldv = self.q.get((state, action), None)
        if oldv is None:
            # If no previous Q-Value exists, then initialize
            # it with the current reward
            self.q[(state, action)] = reward
        else:
            # Update the Q-Value with the weighted sum of old
            # value and the newly found value.
            #
            # Alpha determines how much importance we give to the
            # new value compared to the old value.
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state):
        """Epsilon-Greedy approach for action selection."""
        if random.random() < self.epsilon:
            # With probability epsilon, we select a random action
            action = random.choice(self.actions)
        else:
            # With probability 1-epsilon, we select the action
            # with the highest Q-value
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            # If there are multiple actions with the same Q-Value,
            # then choose randomly among them
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def learn(self, state1, action1, reward, state2):
        """Get the maximum Q-Value for the next state."""
        maxqnew = max([self.getQ(state2, a) for a in self.actions])

        # Learn the Q-Value based on current reward and future
        # expected rewards.
        self.learnQ(state1, action1, reward, reward + self.gamma * maxqnew)


# A utility function to format floating point numbers. Not
# directly related to Q-learning.
def ff(f, n):
    """Format a floating point number to a string with n digits."""
    fs = '{:f}'.format(f)
    if len(fs) < n:
        return ('{:'+n+'s}').format(fs)
    else:
        return fs[:n]
