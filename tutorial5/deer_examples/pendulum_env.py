import numpy as np
import copy

from deer.base_classes import Environment
import gym

class MyEnv(Environment):
    def __init__(self, rng):
        """ Initialize environment.

        Arguments:
            rng - the numpy random number generator            
        """
        # Defining the type of environment
        self.env = gym.make('CartPole-v0')
        self._last_observation = self.env.reset()
        self.is_terminal=False
        self._input_dim = [(1,), (1,), (1,), (1,)]  # self.env.observation_space.shape is equal to 4 
                                                    # and we use only the current value in the belief state

    def act(self, action):
        """ Simulate one time step in the environment.
        """
        
        self._last_observation, reward, self.is_terminal, info = self.env.step(action)
        if (self.mode==0): # Show the policy only at test time
            self.env.render()
            
        return reward
                
    def reset(self, mode=0):
        """ Reset environment for a new episode.

        Arguments:
        Mode : int
            -1 corresponds to training and 0 to test
        """
        # Reset initial observation to a random x and theta
        self._last_observation = self.env.reset()
        self.is_terminal=False
        self.mode=mode

        return self._last_observation
                
    def inTerminalState(self):
        """Tell whether the environment reached a terminal state after the last transition (i.e. the last transition 
        that occured was terminal).
        """
        return self.is_terminal

    def inputDimensions(self):
        return self._input_dim  

    def nActions(self):
        return 2 #Would be useful to have this directly in gym : self.env.action_space.shape  

    def observe(self):
        return copy.deepcopy(self._last_observation)
        
def main():
    rng = np.random.RandomState(123456)
    myenv=MyEnv(rng)

    print (myenv.observe())
    
if __name__ == "__main__":
    main()
