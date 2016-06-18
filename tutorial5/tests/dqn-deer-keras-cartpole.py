'''
Deep Q-learning approach to the cartpole problem
using OpenAI's gym environment.

As part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning

Inspired by https://github.com/VinF/deer

        @author: Victor Mayoral Vilches <victor@erlerobotics.com>

'''
import gym
import random
import pandas
import numpy as np
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, merge, Activation, Convolution2D, MaxPooling2D, Reshape
import theano.tensor as T
import sys
import logging
import numpy as np

from theano import config
import numpy as np

class QNetwork(object):
    """ All the Q-networks classes should inherit this interface.

    Parameters
    -----------
    environment : object from class Environment
        The environment linked to the Q-network
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    """
    def __init__(self, environment, batch_size):
        self._environment = environment
        self._df = 0.9
        self._lr = 0.0002
        self._input_dimensions = self._environment.inputDimensions()
        self._n_actions = self._environment.nActions()
        self._batch_size = batch_size

    def train(self, states, actions, rewards, nextStates, terminals):
        """ This method performs the Bellman iteration for one batch of tuples.
        """
        raise NotImplementedError()

    def chooseBestAction(self, state):
        """ Get the best action for a belief state
        """        
        raise NotImplementedError()

    def qValues(self, state):
        """ Get the q value for one belief state
        """        
        raise NotImplementedError()

    def setLearningRate(self, lr):
        """ Setting the learning rate

        Parameters
        -----------
        lr : float
            The learning rate that has to bet set
        """
        self._lr = lr

    def setDiscountFactor(self, df):
        """ Setting the discount factor

        Parameters
        -----------
        df : float
            The discount factor that has to bet set
        """
        if df < 0. or df > 1.:
            raise AgentError("The discount factor should be in [0,1]")

        self._df = df

    def learningRate(self):
        """ Getting the learning rate
        """
        return self._lr

    def discountFactor(self):
        """ Getting the discount factor
        """
        return self._df


class NN():
    """
    Deep Q-learning network using Keras
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_actions :
    random_state : numpy random number generator
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions

    def _buildDQN(self):
        """
        Build a network consistent with each type of inputs
        """
        layers=[]
        outs_conv=[]
        inputs=[]

        for i, dim in enumerate(self._input_dimensions):
            nfilter=[]
            # - observation[i] is a FRAME
            if len(dim) == 3: #FIXME
                input = Input(shape=(dim[0],dim[1],dim[2]))
                inputs.append(input)
                #reshaped=Reshape((dim[0],dim[1],dim[2]), input_shape=(dim[0],dim[1]))(input)
                x = Convolution2D(32, 8, 8, border_mode='valid')(input)
                x = MaxPooling2D(pool_size=(4, 4), strides=None, border_mode='valid')(x)
                x = Convolution2D(64, 4, 4, border_mode='valid')(x)
                x = MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid')(x)
                x = Convolution2D(64, 3, 3)(x)
                
                out = Flatten()(x)
                
            # - observation[i] is a VECTOR
            elif len(dim) == 2 and dim[0] > 3: #FIXME
                input = Input(shape=(dim[0],dim[1]))
                inputs.append(input)
                reshaped=Reshape((1,dim[0],dim[1]), input_shape=(dim[0],dim[1]))(input)
                x = Convolution2D(16, 2, 1, border_mode='valid')(reshaped)
                x = Convolution2D(16, 2, 2)(x)
                
                out = Flatten()(x)

            # - observation[i] is a SCALAR -
            else:
                if dim[0] > 3:
                    # this returns a tensor
                    input = Input(shape=(dim[0],))
                    inputs.append(input)
                    reshaped=Reshape((1,1,dim[0]), input_shape=(dim[0],))(input)
                    x = Convolution2D(8, 1, 2, border_mode='valid')(reshaped)
                    x = Convolution2D(8, 1, 2)(x)
                    
                    out = Flatten()(x)
                                        
                else:
                    if(len(dim) == 2):
                    # this returns a tensor

                        input = Input(shape=(dim[1],dim[0]))
                        inputs.append(input)
                        out = Flatten()(input)

                    if(len(dim) == 1):
                        input = Input(shape=(dim[0],))
                        inputs.append(input)
                        out=input
                    
            outs_conv.append(out)

        if len(outs_conv)>1:
            x = merge(outs_conv, mode='concat')
        else:
            x= outs_conv [0]
        
        # we stack a deep fully-connected network on top
        x = Dense(50, activation='relu')(x)
        x = Dense(20, activation='relu')(x)
        out = Dense(self._n_actions)(x)

        model = Model(input=inputs, output=out)
        layers=model.layers
        
        # Grab all the parameters together.
        params = [ param
                    for layer in layers 
                    for param in layer.trainable_weights ]

        return model, params

from warnings import warn
from keras.optimizers import SGD,RMSprop

class MyQNetwork(QNetwork):
    """
    Deep Q-learning network using Keras
    
    Parameters
    -----------
    environment : object from class Environment
    rho : float
        Parameter for rmsprop. Default : 0.9
    rms_epsilon : float
        Parameter for rmsprop. Default : 0.0001
    momentum : float
        Default : 0
    clip_delta : float
        Not implemented.
    freeze_interval : int
        Period during which the target network is freezed and after which the target network is updated. Default : 1000
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    network_type : str
        Not used. Default : None
    update_rule: str
        {sgd,rmsprop}. Default : rmsprop
    batch_accumulator : str
        {sum,mean}. Default : sum
    random_state : numpy random number generator
    double_Q : bool, optional
        Activate or not the double_Q learning.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network : object, optional
        default is deer.qnetworks.NN_keras
    """

    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0, clip_delta=0, freeze_interval=1000, batch_size=32, network_type=None, update_rule="rmsprop", batch_accumulator="sum", random_state=np.random.RandomState(), double_Q=False, neural_network=NN):
        """ Initialize environment
        
        """
        QNetwork.__init__(self,environment, batch_size)

        
        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        #self.clip_delta = clip_delta
        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self.update_counter = 0
                
        Q_net = neural_network(self._batch_size, self._input_dimensions, self._n_actions, self._random_state)
        self.q_vals, self.params = Q_net._buildDQN()
        
        if update_rule == 'deepmind_rmsprop':
            warn("The update_rule used is rmsprop")
            update_rule='rmsprop'            
        
        if (update_rule=="sgd"):
            optimizer = SGD(lr=self._lr, momentum=momentum, nesterov=False)
        elif (update_rule=="rmsprop"):
            optimizer = RMSprop(lr=self._lr, rho=self._rho, epsilon=self._rms_epsilon)
        else:
            raise Exception('The update_rule '+update_rule+ 'is not'
                            'implemented.')
        
        self.q_vals.compile(optimizer=optimizer, loss='mse')
       
        self.next_q_vals, self.next_params = Q_net._buildDQN()
        self.next_q_vals.compile(optimizer='rmsprop', loss='mse') #The parameters do not matter since training is done on self.q_vals

        self.q_vals.summary()
        # self.next_q_vals.summary()

        self._resetQHat()


    def toDump(self):
        # FIXME

        return None,None


    def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
        """
        Train one batch.

        1. Set shared variable in states_shared, next_states_shared, actions_shared, rewards_shared, terminals_shared         
        2. perform batch training

        Parameters
        -----------
        states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        actions_val : b x 1 numpy array of integers
        rewards_val : b x 1 numpy array
        next_states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        terminals_val : b x 1 numpy boolean array (currently ignored)


        Returns
        -------
        Average loss of the batch training
        Individual losses for each tuple
        """
        
        if self.update_counter % self._freeze_interval == 0:
            self._resetQHat()
        
        next_q_vals = self.next_q_vals.predict(next_states_val.tolist())
        
        if(self._double_Q==True):
            next_q_vals_current_qnet=self.q_vals.predict(next_states_val.tolist())
            argmax_next_q_vals=np.argmax(next_q_vals_current_qnet, axis=1)
            max_next_q_vals=next_q_vals[np.arange(self._batch_size),argmax_next_q_vals].reshape((-1, 1))
        else:
            max_next_q_vals=np.max(next_q_vals, axis=1, keepdims=True)

        not_terminals=np.ones_like(terminals_val) - terminals_val
        
        target = rewards_val + not_terminals * self._df * max_next_q_vals.reshape((-1))
        
        q_vals=self.q_vals.predict(states_val.tolist())

        # In order to obtain the individual losses, we predict the current Q_vals and calculate the diff
        q_val=q_vals[np.arange(self._batch_size), actions_val.reshape((-1,))]#.reshape((-1, 1))        
        diff = - q_val + target 
        loss_ind=0.5*pow(diff,2)
                
        q_vals[  np.arange(self._batch_size), actions_val.reshape((-1,))  ] = target
                
        # Is it possible to use something more flexible than this? 
        # Only some elements of next_q_vals are actual value that I target. 
        # My loss should only take these into account.
        # Workaround here is that many values are already "exact" in this update
        loss=self.q_vals.train_on_batch(states_val.tolist() , q_vals ) 
                
        self.update_counter += 1        

        return np.sqrt(loss),loss_ind


    def qValues(self, state_val):
        """ Get the q values for one belief state

        Arguments
        ---------
        state_val : one belief state

        Returns
        -------
        The q value for the provided belief state
        """ 
        return self.q_vals.predict([np.expand_dims(state,axis=0) for state in state_val])[0]

    def chooseBestAction(self, state):
        """ Get the best action for a belief state

        Arguments
        ---------
        state : one belief state

        Returns
        -------
        The best action : int
        """        
        q_vals = self.qValues(state)

        return np.argmax(q_vals)
        
    def _resetQHat(self):
        for i,(param,next_param) in enumerate(zip(self.params, self.next_params)):
            next_param.set_value(param.get_value())


from deer.base_classes import Environment
import copy

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
        

import deer.experiment.base_controllers as bc
from deer.default_parser import process_args
from deer.agent import NeuralAgent

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 200
    EPOCHS = 300
    STEPS_PER_TEST = 200
    PERIOD_BTW_SUMMARY_PERFS = 10

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 1

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'sgd'
    BATCH_ACCUMULATOR = 'sum'
    LEARNING_RATE = 0.1
    LEARNING_RATE_DECAY = 0.99
    DISCOUNT = 0.9
    DISCOUNT_INC = 1.
    DISCOUNT_MAX = 0.95
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = 0.2
    EPSILON_DECAY = 10000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "General_DQN_0"
    FREEZE_INTERVAL = 100
    DETERMINISTIC = True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(12345)
    else:
        rng = np.random.RandomState()
    
    # --- Instantiate environment ---
    env = MyEnv(rng)

    # --- Instantiate qnetwork ---
    qnetwork = MyQNetwork(
        env,
        parameters.rms_decay,
        parameters.rms_epsilon,
        parameters.momentum,
        parameters.clip_delta,
        parameters.freeze_interval,
        parameters.batch_size,
        parameters.network_type,
        parameters.update_rule,
        parameters.batch_accumulator,
        rng,
        double_Q=True)
    
    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        qnetwork,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng)

    # --- Bind controllers to the agent ---
    # For comments, please refer to run_toy_env.py
    agent.attach(bc.VerboseController(
        evaluate_on='epoch', 
        periodicity=1))

    agent.attach(bc.TrainerController(
        evaluate_on='action',
        periodicity=parameters.update_frequency, 
        show_episode_avg_V_value=False, 
        show_avg_Bellman_residual=False))

    agent.attach(bc.LearningRateController(
        initial_learning_rate=parameters.learning_rate,
        learning_rate_decay=parameters.learning_rate_decay,
        periodicity=1))

    agent.attach(bc.DiscountFactorController(
        initial_discount_factor=parameters.discount,
        discount_factor_growth=parameters.discount_inc,
        discount_factor_max=parameters.discount_max,
        periodicity=1))

    agent.attach(bc.EpsilonController(
        initial_e=parameters.epsilon_start, 
        e_decays=parameters.epsilon_decay, 
        e_min=parameters.epsilon_min,
        evaluate_on='action', 
        periodicity=1, 
        reset_every='none'))

    agent.attach(bc.InterleavedTestEpochController(
        id=0, 
        epoch_length=parameters.steps_per_test, 
        controllers_to_disable=[0, 1, 2, 3, 4], 
        periodicity=2, 
        show_score=True,
        summarize_every=parameters.period_btw_summary_perfs))
    
    # --- Run the experiment ---
    agent.run(parameters.epochs, parameters.steps_per_epoch)
