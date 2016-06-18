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


# class NN():
#     """
#     Deep Q-learning network using Keras
    
#     Parameters
#     -----------
#     batch_size : int
#         Number of tuples taken into account for each iteration of gradient descent
#     input_dimensions :
#     n_actions :
#     random_state : numpy random number generator
#     """
#     def __init__(self, batch_size, input_dimensions, n_actions, random_state):
#         self._input_dimensions=input_dimensions
#         self._batch_size=batch_size
#         self._random_state=random_state
#         self._n_actions=n_actions

#     def _buildDQN(self):
#         """
#         Build a network consistent with each type of inputs
#         """
#         layers=[]
#         outs_conv=[]
#         inputs=[]

#         for i, dim in enumerate(self._input_dimensions):
#             nfilter=[]
#             # - observation[i] is a FRAME
#             if len(dim) == 3: #FIXME
#                 input = Input(shape=(dim[0],dim[1],dim[2]))
#                 inputs.append(input)
#                 #reshaped=Reshape((dim[0],dim[1],dim[2]), input_shape=(dim[0],dim[1]))(input)
#                 x = Convolution2D(32, 8, 8, border_mode='valid')(input)
#                 x = MaxPooling2D(pool_size=(4, 4), strides=None, border_mode='valid')(x)
#                 x = Convolution2D(64, 4, 4, border_mode='valid')(x)
#                 x = MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid')(x)
#                 x = Convolution2D(64, 3, 3)(x)
                
#                 out = Flatten()(x)
                
#             # - observation[i] is a VECTOR
#             elif len(dim) == 2 and dim[0] > 3: #FIXME
#                 input = Input(shape=(dim[0],dim[1]))
#                 inputs.append(input)
#                 reshaped=Reshape((1,dim[0],dim[1]), input_shape=(dim[0],dim[1]))(input)
#                 x = Convolution2D(16, 2, 1, border_mode='valid')(reshaped)
#                 x = Convolution2D(16, 2, 2)(x)
                
#                 out = Flatten()(x)

#             # - observation[i] is a SCALAR -
#             else:
#                 if dim[0] > 3:
#                     # this returns a tensor
#                     input = Input(shape=(dim[0],))
#                     inputs.append(input)
#                     reshaped=Reshape((1,1,dim[0]), input_shape=(dim[0],))(input)
#                     x = Convolution2D(8, 1, 2, border_mode='valid')(reshaped)
#                     x = Convolution2D(8, 1, 2)(x)
                    
#                     out = Flatten()(x)
                                        
#                 else:
#                     if(len(dim) == 2):
#                     # this returns a tensor

#                         input = Input(shape=(dim[1],dim[0]))
#                         inputs.append(input)
#                         out = Flatten()(input)

#                     if(len(dim) == 1):
#                         input = Input(shape=(dim[0],))
#                         inputs.append(input)
#                         out=input
                    
#             outs_conv.append(out)

#         if len(outs_conv)>1:
#             x = merge(outs_conv, mode='concat')
#         else:
#             x= outs_conv [0]
        
#         # we stack a deep fully-connected network on top
#         x = Dense(50, activation='relu')(x)
#         x = Dense(20, activation='relu')(x)
#         out = Dense(self._n_actions)(x)

#         model = Model(input=inputs, output=out)
#         layers=model.layers
        
#         # Grab all the parameters together.
#         params = [ param
#                     for layer in layers 
#                     for param in layer.trainable_weights ]

#         return model, params

# from warnings import warn
# from keras.optimizers import SGD,RMSprop

# class MyQNetwork(QNetwork):
#     """
#     Deep Q-learning network using Keras
    
#     Parameters
#     -----------
#     environment : object from class Environment
#     rho : float
#         Parameter for rmsprop. Default : 0.9
#     rms_epsilon : float
#         Parameter for rmsprop. Default : 0.0001
#     momentum : float
#         Default : 0
#     clip_delta : float
#         Not implemented.
#     freeze_interval : int
#         Period during which the target network is freezed and after which the target network is updated. Default : 1000
#     batch_size : int
#         Number of tuples taken into account for each iteration of gradient descent. Default : 32
#     network_type : str
#         Not used. Default : None
#     update_rule: str
#         {sgd,rmsprop}. Default : rmsprop
#     batch_accumulator : str
#         {sum,mean}. Default : sum
#     random_state : numpy random number generator
#     double_Q : bool, optional
#         Activate or not the double_Q learning.
#         More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
#     neural_network : object, optional
#         default is deer.qnetworks.NN_keras
#     """

#     def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0, clip_delta=0, freeze_interval=1000, batch_size=32, network_type=None, update_rule="rmsprop", batch_accumulator="sum", random_state=np.random.RandomState(), double_Q=False, neural_network=NN):
#         """ Initialize environment
        
#         """
#         QNetwork.__init__(self,environment, batch_size)

        
#         self._rho = rho
#         self._rms_epsilon = rms_epsilon
#         self._momentum = momentum
#         #self.clip_delta = clip_delta
#         self._freeze_interval = freeze_interval
#         self._double_Q = double_Q
#         self._random_state = random_state
#         self.update_counter = 0
                
#         Q_net = neural_network(self._batch_size, self._input_dimensions, self._n_actions, self._random_state)
#         self.q_vals, self.params = Q_net._buildDQN()
        
#         if update_rule == 'deepmind_rmsprop':
#             warn("The update_rule used is rmsprop")
#             update_rule='rmsprop'            
        
#         if (update_rule=="sgd"):
#             optimizer = SGD(lr=self._lr, momentum=momentum, nesterov=False)
#         elif (update_rule=="rmsprop"):
#             optimizer = RMSprop(lr=self._lr, rho=self._rho, epsilon=self._rms_epsilon)
#         else:
#             raise Exception('The update_rule '+update_rule+ 'is not'
#                             'implemented.')
        
#         self.q_vals.compile(optimizer=optimizer, loss='mse')
       
#         self.next_q_vals, self.next_params = Q_net._buildDQN()
#         self.next_q_vals.compile(optimizer='rmsprop', loss='mse') #The parameters do not matter since training is done on self.q_vals

#         self._resetQHat()


#     def toDump(self):
#         # FIXME

#         return None,None


#     def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
#         """
#         Train one batch.

#         1. Set shared variable in states_shared, next_states_shared, actions_shared, rewards_shared, terminals_shared         
#         2. perform batch training

#         Parameters
#         -----------
#         states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
#         actions_val : b x 1 numpy array of integers
#         rewards_val : b x 1 numpy array
#         next_states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
#         terminals_val : b x 1 numpy boolean array (currently ignored)


#         Returns
#         -------
#         Average loss of the batch training
#         Individual losses for each tuple
#         """
        
#         if self.update_counter % self._freeze_interval == 0:
#             self._resetQHat()
        
#         next_q_vals = self.next_q_vals.predict(next_states_val.tolist())
        
#         if(self._double_Q==True):
#             next_q_vals_current_qnet=self.q_vals.predict(next_states_val.tolist())
#             argmax_next_q_vals=np.argmax(next_q_vals_current_qnet, axis=1)
#             max_next_q_vals=next_q_vals[np.arange(self._batch_size),argmax_next_q_vals].reshape((-1, 1))
#         else:
#             max_next_q_vals=np.max(next_q_vals, axis=1, keepdims=True)

#         not_terminals=np.ones_like(terminals_val) - terminals_val
        
#         target = rewards_val + not_terminals * self._df * max_next_q_vals.reshape((-1))
        
#         q_vals=self.q_vals.predict(states_val.tolist())

#         # In order to obtain the individual losses, we predict the current Q_vals and calculate the diff
#         q_val=q_vals[np.arange(self._batch_size), actions_val.reshape((-1,))]#.reshape((-1, 1))        
#         diff = - q_val + target 
#         loss_ind=0.5*pow(diff,2)
                
#         q_vals[  np.arange(self._batch_size), actions_val.reshape((-1,))  ] = target
                
#         # Is it possible to use something more flexible than this? 
#         # Only some elements of next_q_vals are actual value that I target. 
#         # My loss should only take these into account.
#         # Workaround here is that many values are already "exact" in this update
#         loss=self.q_vals.train_on_batch(states_val.tolist() , q_vals ) 
                
#         self.update_counter += 1        

#         return np.sqrt(loss),loss_ind


#     def qValues(self, state_val):
#         """ Get the q values for one belief state

#         Arguments
#         ---------
#         state_val : one belief state

#         Returns
#         -------
#         The q value for the provided belief state
#         """ 
#         return self.q_vals.predict([np.expand_dims(state,axis=0) for state in state_val])[0]

#     def chooseBestAction(self, state):
#         """ Get the best action for a belief state

#         Arguments
#         ---------
#         state : one belief state

#         Returns
#         -------
#         The best action : int
#         """        
#         q_vals = self.qValues(state)

#         return np.argmax(q_vals)
        
#     def _resetQHat(self):
#         for i,(param,next_param) in enumerate(zip(self.params, self.next_params)):
#             next_param.set_value(param.get_value())


import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class ConvolutionalLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(1, 1), stride=(1,1)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True, st=stride)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


import numpy as np
import theano
import theano.tensor as T

    
class NN():
    """
    Deep Q-learning network using Theano
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_Actions :
    random_state : numpy random number generator
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        
    def _buildDQN(self, inputs):
        """
        Build a network consistent with each type of inputs
        """
        layers=[]
        outs_conv=[]
        outs_conv_shapes=[]
        
        for i, dim in enumerate(self._input_dimensions):
            nfilter=[]
            
            # - observation[i] is a FRAME -
            if len(dim) == 3: 

                ### First layer
                newR = dim[1]
                newC = dim[2]
                fR=8  # filter Rows
                fC=8  # filter Column
                pR=1  # pool Rows
                pC=1  # pool Column
                nfilter.append(32)
                stride_size=4
                l_conv1 = ConvolutionalLayer(
                    rng=self._random_state,
                    input=inputs[i].reshape((self._batch_size,dim[0],newR,newC)),
                    filter_shape=(nfilter[0],dim[0],fR,fC),
                    image_shape=(self._batch_size,dim[0],newR,newC),
                    poolsize=(pR,pC),
                    stride=(stride_size,stride_size)
                )
                layers.append(l_conv1)

                newR = (newR - fR + 1 - pR) // stride_size + 1
                newC = (newC - fC + 1 - pC) // stride_size + 1

                ### Second layer
                fR=4  # filter Rows
                fC=4  # filter Column
                pR=1  # pool Rows
                pC=1  # pool Column
                nfilter.append(64)
                stride_size=2
                l_conv2 = ConvolutionalLayer(
                    rng=self._random_state,
                    input=l_conv1.output.reshape((self._batch_size,nfilter[0],newR,newC)),
                    filter_shape=(nfilter[1],nfilter[0],fR,fC),
                    image_shape=(self._batch_size,nfilter[0],newR,newC),
                    poolsize=(pR,pC),
                    stride=(stride_size,stride_size)
                )
                layers.append(l_conv2)

                newR = (newR - fR + 1 - pR) // stride_size + 1
                newC = (newC - fC + 1 - pC) // stride_size + 1

                ### Third layer
                fR=3  # filter Rows
                fC=3  # filter Column
                pR=1  # pool Rows
                pC=1  # pool Column
                nfilter.append(64)
                stride_size=1
                l_conv3 = ConvolutionalLayer(
                    rng=self._random_state,
                    input=l_conv2.output.reshape((self._batch_size,nfilter[1],newR,newC)),
                    filter_shape=(nfilter[2],nfilter[1],fR,fC),
                    image_shape=(self._batch_size,nfilter[1],newR,newC),
                    poolsize=(pR,pC),
                    stride=(stride_size,stride_size)
                )
                layers.append(l_conv3)

                newR = (newR - fR + 1 - pR) // stride_size + 1
                newC = (newC - fC + 1 - pC) // stride_size + 1

                outs_conv.append(l_conv3.output)
                outs_conv_shapes.append((nfilter[2],newR,newC))

                
            # - observation[i] is a VECTOR -
            elif len(dim) == 2 and dim[0] > 3:                
                
                newR = dim[0]
                newC = dim[1]
                
                fR=2  # filter Rows
                fC=1  # filter Column
                pR=1  # pool Rows
                pC=1  # pool Column
                nfilter.append(16)
                stride_size=1

                l_conv1 = ConvolutionalLayer(
                    rng=self._random_state,
                    input=inputs[i].reshape((self._batch_size,1,newR,newC)),
                    filter_shape=(nfilter[0],1,fR,fC),
                    image_shape=(self._batch_size,1,newR,newC),
                    poolsize=(pR,pC),
                    stride=(stride_size,stride_size)
                )                
                layers.append(l_conv1)
                
                newR = (newR - fR + 1 - pR) // stride_size + 1  # stride 2
                newC = (newC - fC + 1 - pC) // stride_size + 1  # stride 2

                fR=2  # filter Rows
                fC=2  # filter Column
                pR=1  # pool Rows
                pC=1  # pool Column
                nfilter.append(16)
                stride_size=1

                l_conv2 = ConvolutionalLayer(
                    rng=self._random_state,
                    input=l_conv1.output.reshape((self._batch_size,nfilter[0],newR,newC)),
                    filter_shape=(nfilter[1],nfilter[0],fR,fC),
                    image_shape=(self._batch_size,nfilter[0],newR,newC),
                    poolsize=(pR,pC),
                    stride=(stride_size,stride_size)
                )                
                layers.append(l_conv2)
                
                newR = (newR - fR + 1 - pR) // stride_size + 1  # stride 2
                newC = (newC - fC + 1 - pC) // stride_size + 1  # stride 2

                outs_conv.append(l_conv2.output)
                outs_conv_shapes.append((nfilter[1],newR,newC))


            # - observation[i] is a SCALAR -
            else:
                if dim[0] > 3:
                    newR = 1
                    newC = dim[0]
                    
                    fR=1  # filter Rows
                    fC=2  # filter Column
                    pR=1  # pool Rows
                    pC=1  # pool Column
                    nfilter.append(8)
                    stride_size=1

                    l_conv1 = ConvolutionalLayer(
                        rng=self._random_state,
                        input=inputs[i].reshape((self._batch_size,1,newR,newC)),
                        filter_shape=(nfilter[0],1,fR,fC),
                        image_shape=(self._batch_size,1,newR,newC),
                        poolsize=(pR,pC),
                        stride=(stride_size,stride_size)
                    )                
                    layers.append(l_conv1)
                    
                    newC = (newC - fC + 1 - pC) // stride_size + 1  # stride 2

                    fR=1  # filter Rows
                    fC=2  # filter Column
                    pR=1  # pool Rows
                    pC=1  # pool Column
                    nfilter.append(8)
                    stride_size=1
                    
                    l_conv2 = ConvolutionalLayer(
                        rng=self._random_state,
                        input=l_conv1.output.reshape((self._batch_size,nfilter[0],newR,newC)),
                        filter_shape=(nfilter[1],nfilter[0],fR,fC),
                        image_shape=(self._batch_size,nfilter[0],newR,newC),
                        poolsize=(pR,pC),
                        stride=(stride_size,stride_size)
                    )                
                    layers.append(l_conv2)
                    
                    newC = (newC - fC + 1 - pC) // stride_size + 1  # stride 2

                    outs_conv.append(l_conv2.output)
                    outs_conv_shapes.append((nfilter[1],newC))
                    
                else:
                    if(len(dim) == 2):
                        outs_conv_shapes.append((dim[0],dim[1]))
                    elif(len(dim) == 1):
                        outs_conv_shapes.append((1,dim[0]))
                    outs_conv.append(inputs[i])
        
        
        ## Custom merge of layers
        output_conv = outs_conv[0].flatten().reshape((self._batch_size, np.prod(outs_conv_shapes[0])))
        shapes=np.prod(outs_conv_shapes[0])

        if (len(outs_conv)>1):
            for out_conv,out_conv_shape in zip(outs_conv[1:],outs_conv_shapes[1:]):
                output_conv=T.concatenate((output_conv, out_conv.flatten().reshape((self._batch_size, np.prod(out_conv_shape)))) , axis=1)
                shapes+=np.prod(out_conv_shape)
                shapes

                
        self.hiddenLayer1 = HiddenLayer(rng=self._random_state, input=output_conv,
                                       n_in=shapes, n_out=50,
                                       activation=T.tanh)                                       
        layers.append(self.hiddenLayer1)

        self.hiddenLayer2 = HiddenLayer(rng=self._random_state, input=self.hiddenLayer1.output,
                                       n_in=50, n_out=20,
                                       activation=T.tanh)
        layers.append(self.hiddenLayer2)

        self.outLayer = HiddenLayer(rng=self._random_state, input=self.hiddenLayer2.output,
                                       n_in=20, n_out=self._n_actions,
                                       activation=None)
        layers.append(self.outLayer)

        # Grab all the parameters together.
        params = [param
                       for layer in layers
                       for param in layer.params]
        
        return self.outLayer.output, params, outs_conv_shapes

import theano
import theano.tensor as T
from collections import OrderedDict
import numpy as np



def deepmind_rmsprop(loss_or_grads, params, grads, learning_rate, 
                     rho, epsilon):
    """RMSProp updates [1]_.

    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    rho : float or symbolic scalar
        Gradient moving average decay factor
    epsilon : float or symbolic scalar
        Small value added for numerical stability

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.

    Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
    learning rate :math:`\\eta_t` is calculated as:

    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}

    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """

    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)

        acc_grad = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        acc_grad_new = rho * acc_grad + (1 - rho) * grad

        acc_rms = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        acc_rms_new = rho * acc_rms + (1 - rho) * grad ** 2


        updates[acc_grad] = acc_grad_new
        updates[acc_rms] = acc_rms_new

        updates[param] = (param - learning_rate * 
                          (grad / 
                           T.sqrt(acc_rms_new - acc_grad_new **2 + epsilon)))

    return updates


import numpy as numpy
import theano
import theano.tensor as T
    
class MyQNetwork(QNetwork):
    """
    Deep Q-learning network using Theano
    
    Parameters
    -----------
    environment : object from class Environment
    rho : float
        Parameter for rmsprop. Default : 0.9
    rms_epsilon : float
        Parameter for rmsprop. Default : 0.0001
    momentum : float
        Not implemented. Default : 0
    clip_delta : float
        If > 0, the squared loss is linear past the clip point which keeps the gradient constant. Default : 0
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
        Default : random seed.
    double_Q : bool
        Activate or not the DoubleQ learning : not implemented yet. Default : False
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network  : object
        default is deer.qnetworks.NN_theano
    """

    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0, clip_delta=0, freeze_interval=1000, batch_size=32, network_type=None, update_rule="rmsprop", batch_accumulator="sum", random_state=np.random.RandomState(), double_Q=False, neural_network=NN):
        """ Initialize environment
        
        """
        QNetwork.__init__(self,environment, batch_size)
        
        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._clip_delta = clip_delta
        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        
        self.update_counter = 0
        
        states=[]   # list of symbolic variables for each of the k element in the belief state
                    # --> [ T.tensor4 if observation of element=matrix, T.tensor3 if vector, T.tensor 2 if scalar ]
        next_states=[] # idem than states at t+1 
        self.states_shared=[] # list of shared variable for each of the k element in the belief state
        self.next_states_shared=[] # idem that self.states_shared at t+1

        for i, dim in enumerate(self._input_dimensions):
            if len(dim) == 3:
                states.append(T.tensor4("%s_%s" % ("state", i)))
                next_states.append(T.tensor4("%s_%s" % ("next_state", i)))

            elif len(dim) == 2:
                states.append(T.tensor3("%s_%s" % ("state", i)))
                next_states.append(T.tensor3("%s_%s" % ("next_state", i)))
                
            elif len(dim) == 1:            
                states.append( T.matrix("%s_%s" % ("state", i)) )
                next_states.append( T.matrix("%s_%s" % ("next_state", i)) )
                
            self.states_shared.append(theano.shared(np.zeros((batch_size,) + dim, dtype=theano.config.floatX) , borrow=False))
            self.next_states_shared.append(theano.shared(np.zeros((batch_size,) + dim, dtype=theano.config.floatX) , borrow=False))
        
        print("Number of observations per state: {}".format(len(self.states_shared)))
        print("For each observation, historySize + ponctualObs_i.shape: {}".format(self._input_dimensions))
                
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')
        thediscount = T.scalar(name='thediscount', dtype=theano.config.floatX)
        thelr = T.scalar(name='thelr', dtype=theano.config.floatX)
        
        Q_net=neural_network(self._batch_size, self._input_dimensions, self._n_actions, self._random_state)
        self.q_vals, self.params, shape_after_conv = Q_net._buildDQN(states)
        
        print("Number of neurons after spatial and temporal convolution layers: {}".format(shape_after_conv))

        self.next_q_vals, self.next_params, shape_after_conv = Q_net._buildDQN(next_states)
        self._resetQHat()

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        
        
        if(self._double_Q==True):
            givens_next={}
            for i, x in enumerate(self.next_states_shared):
                givens_next[ states[i] ] = x

            self.next_q_vals_current_qnet=theano.function([], self.q_vals,
                                          givens=givens_next)

            next_q_curr_qnet = theano.clone(self.next_q_vals)

            argmax_next_q_vals=T.argmax(next_q_curr_qnet, axis=1, keepdims=True)

            max_next_q_vals=self.next_q_vals[T.arange(batch_size),argmax_next_q_vals.reshape((-1,))].reshape((-1, 1))

        else:
            max_next_q_vals=T.max(self.next_q_vals, axis=1, keepdims=True)


        not_terminals=T.ones_like(terminals) - terminals

        target = rewards + not_terminals * thediscount * max_next_q_vals

        q_val=self.q_vals[T.arange(batch_size), actions.reshape((-1,))].reshape((-1, 1))
        # Note : Strangely (target - q_val) lead to problems with python 3.5, theano 0.8.0rc and floatX=float32...
        diff = - q_val + target 

        if self._clip_delta > 0:
            # This loss function implementation is taken from
            # https://github.com/spragunr/deep_q_rl
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self._clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss_ind = 0.5 * quadratic_part ** 2 + self._clip_delta * linear_part
        else:
            loss_ind = 0.5 * diff ** 2

        if batch_accumulator == 'sum':
            loss = T.sum(loss_ind)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss_ind)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        givens = {
            rewards: self.rewards_shared,
            actions: self.actions_shared, ## actions not needed!
            terminals: self.terminals_shared
        }
        
        for i, x in enumerate(self.states_shared):
            givens[ states[i] ] = x 
        for i, x in enumerate(self.next_states_shared):
            givens[ next_states[i] ] = x
                
                
        gparams=[]
        for p in self.params:
            gparam =  T.grad(loss, p)
            gparams.append(gparam)

        updates = []
        
        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, self.params, gparams, thelr, self._rho,
                                       self._rms_epsilon)
        elif update_rule == 'rmsprop':
            for i,(p, g) in enumerate(zip(self.params, gparams)):                
                acc = theano.shared(p.get_value() * 0.)
                acc_new = rho * acc + (1 - self._rho) * g ** 2
                gradient_scaling = T.sqrt(acc_new + self._rms_epsilon)
                g = g / gradient_scaling
                updates.append((acc, acc_new))
                updates.append((p, p - thelr * g))

        elif update_rule == 'sgd':
            for i, (param, gparam) in enumerate(zip(self.params, gparams)):
                updates.append((param, param - thelr * gparam))
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))
    
        
        if(self._double_Q==True):
            self._train = theano.function([thediscount, thelr, next_q_curr_qnet], [loss, loss_ind, self.q_vals], updates=updates,
                                      givens=givens,
                                      on_unused_input='warn')
        else:
            self._train = theano.function([thediscount, thelr], [loss, loss_ind, self.q_vals], updates=updates,
                                      givens=givens,
                                      on_unused_input='warn')
        givens2={}
        for i, x in enumerate(self.states_shared):
            givens2[ states[i] ] = x 

        self._q_vals = theano.function([], self.q_vals,
                                      givens=givens2,
                                      on_unused_input='warn')

            
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
        average loss of the batch training
        individual losses for each tuple
        """
        
        for i in range(len(self.states_shared)):
            self.states_shared[i].set_value(states_val[i])
            
        for i in range(len(self.states_shared)):
            self.next_states_shared[i].set_value(next_states_val[i])

        
        self.actions_shared.set_value(actions_val.reshape(len(actions_val), 1))
        self.rewards_shared.set_value(rewards_val.reshape(len(rewards_val), 1))
        self.terminals_shared.set_value(terminals_val.reshape(len(terminals_val), 1))
        if self.update_counter % self._freeze_interval == 0:
            self._resetQHat()
        
        if(self._double_Q==True):
            self._next_q_curr_qnet = self.next_q_vals_current_qnet()
            loss, loss_ind, _ = self._train(self._df, self._lr,self._next_q_curr_qnet)
        else:
            loss, loss_ind, _ = self._train(self._df, self._lr)

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
        # Set the first element of the batch to values provided by state_val
        for i in range(len(self.states_shared)):
            aa = self.states_shared[i].get_value()
            aa[0] = state_val[i]
            self.states_shared[i].set_value(aa)
        
        return self._q_vals()[0]

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
    EPOCHS = 200
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
    LEARNING_RATE = 0.01
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
