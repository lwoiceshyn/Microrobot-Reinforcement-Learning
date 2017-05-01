import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math

from workspace import *

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
# from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

env = WorkSpace(dt=0.05)
env.reset(preset='A')

# Setting up the Neural Network Agent Representation

# Hyperparameters
H = 10 # Hidden Layer Neurons
H2 = 10
batch_size = 5 
learning_rate = 0.05
gamma = 0.95 # Discount Factor for Reward

D = 8 # Input Dimensionality

tf.reset_default_graph()

#This defines the network as it performs a mapping from from an observation of the environment to 
#a probability of choosing the action of moving up, down, left, or right.
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D,H], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H,H2], initializer=tf.contrib.layers.xavier_initializer())
layer2 = tf.nn.relu(tf.matmul(layer1, W2))
W3 = tf.get_variable("W3", shape=[H2,1], initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer2, W3)
probability = tf.nn.sigmoid(score)

# Components of network used to learn the reinforcement learning policy
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# The loss function. This sends the weights in the direction of making actions 
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# Gradients are not updated after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # our optimizer
W1Grad = tf.placeholder(tf.float32, name="batch_grad1") # Placeholder to send the final gradients through when we update
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

# Advantage Function
def discount_rewards(r):
    ''' Takes a 1D float array of rewards and computes discounted reward '''
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = 0
reward_sum = 0
episode_number = 1
total_episodes = 10000
actions = []
init = tf.global_variables_initializer()

#Launches the Tensorflow Graph
with tf.Session() as sess:
    rendering = False # CHANGE BACK TO FALSE
    sess.run(init)
    observation = env.reset(preset='A') # Obtains an initial observation of the environment

    # Reset the gradient placeholder. We will collect gradients in 
    # gradBuffer until we are ready to update our policy network. 
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:

        # Rendering the environment slows things down, 
        # so let's only look at it once our agent is doing a good job.
        if running_reward/batch_size > 199 or rendering == True:
            env.render()
            rendering = True
        
        # Make sure the observation is in a shape the network can handle
        x = np.reshape(observation, [1,D])

        # Run the policy network and get an action to take
        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0
        actions.append(action)

        xs.append(x) # observation
        y = 1 if action == 0 else 0  # a "fake label"
        ys.append(y)

        # Step the environment and get new measurements
        # print(action)
        observation, reward, done = env.step(action, dt=0.1)
        reward_sum += reward

        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done:
            # print(np.mean(actions))
            actions = []
            episode_number += 1
            # stack together all the inputs, hidden states, action gradients for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], [] # reset array memory

            # compute the discounted reward backwards through Time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # get the gradient for this episode and save it in the gradBuffer
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad
            
            # if we have completed enough episodes, then update the policy network with our gradients
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # summary
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print("Average reward for episode %d is: %f. Total average reward %f." % (episode_number, reward_sum/batch_size, running_reward/batch_size))

                if running_reward/batch_size > 500:
                    print("Task solved in %d episodes!" % episode_number)
                    break
                
                reward_sum = 0

            observation = env.reset(preset='A')

print(episode_number, 'Episodes completed.')

