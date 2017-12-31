import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt

"""
Vanilla Policy Gradient (PG)

- Sources of inspiration:
    - https://github.com/kvfrans/openai-cartpole/blob/master/cartpole-policygradient.py
    - http://kvfrans.com/simple-algoritms-for-solving-cartpole/
    - https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
    - https://gist.github.com/greydanus/7cef68683ec955720ddde6b3edf8820e
    - https://gist.github.com/cgnicholls/0127c885dbff07fde8dc1d7bfe62ac1a
    - https://gist.github.com/mohakbhardwaj/1cebbf58d4c0627c9335d9b7eb55b803
    - https://theneuralperspective.com/2016/11/26/1656/
    - https://theneuralperspective.com/2016/11/25/reinforcement-learning-rl-policy-gradients-i/
    - https://gist.github.com/cgnicholls/0127c885dbff07fde8dc1d7bfe62ac1a
    - https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient
    - https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb

- Paper: http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
- Pseudocode (vanilla PG):
1. Initialize policy (e.g. NNs) parameter $\theta$ and baseline $b$
2. For iteration=1,2,... do
    2.1 Collect a set of trajectories by executing the current policy obtaining
        $\mathbf{s}_{0:H},\mathbf{a}_{0:H},r_{0:H}$
    2.2 At each timestep in each trajectory, compute
        2.2.1 the return $R_t = \sum_{t'=t}^{T-1} \gamma^{t'-t}r_{t'}$ and
        2.2.2 the advantage estimate $\hat{A_t} = R_t - b(s_t)$.
    2.3 Re-fit the baseline (recomputing the value function) by minimizing
        $|| b(s_t) - R_t||^2$, summed over all trajectories and timesteps.

    2.3 Re-fit the baseline (recomputing the value function) by minimizing
        $|| b(s_t) - R_t||^2$, summed over all trajectories and timesteps.
        In other words, estimate optimal baseline:

          $b=\frac{\left\langle \left(  \sum\nolimits_{h=0}^{H} \mathbf{\nabla}_{\theta_{k}}\log\pi_{\mathbf{\theta}}
          \left(  \mathbf{a}_{h}\left\vert \mathbf{s}_{h}\right.  \right)  \right)  ^{2}\sum\nolimits_{l=0}^{H}
          \gamma r_{l}\right\rangle }{\left\langle \left(
          \sum\nolimits_{h=0}^{H}\mathbf{\nabla}_{\theta_{k}}\log\pi_{\mathbf{\theta}
          }\left(  \mathbf{a}_{h}\left\vert \mathbf{x}_{h}\right.  \right)  \right)
          ^{2}\right\rangle }$

    2.4 Update the policy, using a policy gradient estimate $\hat{g}$,
        which is a sum of terms
            $\nabla_\theta log\pi(a_t | s_t,\theta)\hat(A_t)$
3. **end for**
"""

def softmax(x):
    """
    Softmax function implementation
    """
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def policy_gradient():
    """
    Implementation of an optimizer that allows to incrementally update
    the policy.

    Parameters
    ----------
    None

    Returns
    -------
    probabilities
        tensor containing the probabilities
    state
        tensor containing the state of the environment
    actions
        one-hot encoded vector containing a "one" at the action
        we want to increase probability of.
    advantages
        tensor with the advantage from the environment
    optimizer
        Adam optimizer
    """
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters",[4,2])
        state = tf.placeholder("float",[None,4])
        actions = tf.placeholder("float",[None,2])
        advantages = tf.placeholder("float",[None,1])
        linear = tf.matmul(state,params)
        probabilities = tf.nn.softmax(linear)
        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions),reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        # maximize the log probability
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probabilities, state, actions, advantages, optimizer

def value_gradient():
    """
    Define the value network (the critic) which returns a value
    for each state. Critic is implemented as a single hidden layer
    neural network (with 10 hidden neurons).

    Returns
    -------
    calculated
        tensor containing the probabilities
    state
        tensor containing the state of the environment
    newvals
        tensor containing updated value function outputs (for optimization)
    optimizer
        Adam optimizer
    loss
        loss tensor. Debug purposes.
    """
    with tf.variable_scope("value"):
        state = tf.placeholder("float",[None,4])
        newvals = tf.placeholder("float",[None,1])
        w1 = tf.get_variable("w1",[4,10])
        b1 = tf.get_variable("b1",[10])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[10,1])
        b2 = tf.get_variable("b2",[1])
        calculated = tf.matmul(h1,w2) + b2
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess, render = False):
    """
    This function implements the main part of PG

    Params
    -------
    env
        Environment where to use PG
    policy_grad
        Policy gradient function that a) calculates the policy and b) optimizes it
    value_grad
        Value network (the critic)
    sess
        TF session object

    Returns
    -------
    totalreward
        the total reward of the episode (200 steps)

    """
    # initialize variables
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []

                            # 2.1 Collect a set of trajectories by executing the current policy obtaining
                            #     $\mathbf{s}_{0:H},\mathbf{a}_{0:H},r_{0:H}$
    for _ in range(200):
        if render:
            env.render()
        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated,feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0,1) < probs[0][0] else 1
        # record the transition
        states.append(observation)
        actionblank = np.zeros(2) # hardcode to specific environment
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break
                            # 2.2 At each timestep in each trajectory, compute
    for index, trans in enumerate(transitions):
        # invidivual transition
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in range(future_transitions):
                            # 2.2.1 the return $R_t = \sum_{t'=t}^{T-1} \gamma^{t'-t}r_{t'}$
            future_reward += transitions[(index2) + index][2] * decrease
            # here "decrease" represents the gamma
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated,feed_dict={vl_state: obs_vector})[0][0]
                            # 2.2.2 the advantage estimate $\hat{A_t} = R_t - b(s_t)$.
        # advantage: how much better was this action than normal
        advantages.append(future_reward - currentval)

        # add future_reward to update the value function towards new return
        update_vals.append(future_reward)

                            # 2.3 Re-fit the baseline (recomputing the value function) by minimizing
                            #     $|| b(s_t) - R_t||^2$, summed over all trajectories and timesteps.
                            #     In other words, estimate optimal baseline:
                            #
                            #       $b=\frac{\left\langle \left(  \sum\nolimits_{h=0}^{H} \mathbf{\nabla}_{\theta_{k}}\log\pi_{\mathbf{\theta}}
                            #       \left(  \mathbf{a}_{h}\left\vert \mathbf{s}_{h}\right.  \right)  \right)  ^{2}\sum\nolimits_{l=0}^{H}
                            #       \gamma r_{l}\right\rangle }{\left\langle \left(
                            #       \sum\nolimits_{h=0}^{H}\mathbf{\nabla}_{\theta_{k}}\log\pi_{\mathbf{\theta}
                            #       }\left(  \mathbf{a}_{h}\left\vert \mathbf{x}_{h}\right.  \right)  \right)
                            #       ^{2}\right\rangle }$
    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

                            # 2.4 Update the policy, using a policy gradient estimate $\hat{g}$,
                            #     which is a sum of terms $\nabla_\theta log\pi(a_t | s_t,\theta)\hat(A_t)$.
                            #     In other words:
                            #
                            #       $g_{k}=\left\langle \left(  \sum\nolimits_{h=0}^{H}\mathbf{\nabla
                            #       }_{\theta_{k}}\log\pi_{\mathbf{\theta}}\left(  \mathbf{a}_{h}\left\vert
                            #       \mathbf{s}_{h}\right.  \right)  \right)  \left(  \sum\nolimits_{l=0}^{H}
                            #       \gamma r_{l}-b\right)  \right\rangle$
    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    return totalreward


env = gym.make('CartPole-v0')
                            # 1. Initialize policy (e.g. NNs) parameter $\theta$ and baseline $b$
policy_grad = policy_gradient()
value_grad = value_gradient()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
                            # 2. For iteration=1,2,... do
for i in range(1000):
    reward = run_episode(env, policy_grad, value_grad, sess)
    print("episode ",i, "reward: ",reward)
    # if reward == 200:
    #     print("reward 200")
    #     print(i)
    #     break
                            # 3. **end for**

# Validate the training in 1000 epidodes
t = 0
for _ in range(1000):
    reward = run_episode(env, policy_grad, value_grad, sess, render=True)
    t += reward
print(t / 1000)
