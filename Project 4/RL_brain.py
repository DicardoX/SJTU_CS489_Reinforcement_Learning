# RL_brain.py
# Ref: https://github.com/cristianoc20/RL_learning/blob/master/Open_AI_gym/RL_brain.py
# Implementation of DeepQNetwork Class

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


# Deep Q Network Off-policy
class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 epsilon_greedy=0.9,
                 replace_target_iter=300,
                 buffer_size=500,
                 batch_size=32,
                 epsilon_greedy_increment=None,
                 output_graph=False
                 ):
        self.n_actions = n_actions
        # Each state can be described as n_features features list
        self.n_features = n_features
        self.lr = learning_rate
        self._gamma = reward_decay
        self._epsilon_max = epsilon_greedy
        # Update Q target each self.replace_target_iter rounds
        self.replace_target_iter = replace_target_iter
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self._epsilon_increment = epsilon_greedy_increment
        self._epsilon = 0 if epsilon_greedy_increment is not None else self._epsilon_max
        # Total learn step
        self.learn_step_counter = 0
        # Initialize memory buffer
        self.buffer = np.zeros((self.buffer_size, n_features * 2 + 2))

        # Consist of [Target_net, Evaluate_net]
        self._build_net()
        target_params = tf.get_collection("target_net_params")
        evaluate_params = tf.get_collection("eval_net_params")
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(target_params, evaluate_params)]

        self.session = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.session.graph)

        self.session.run(tf.global_variables_initializer())
        self.cost_history = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        # Input for evaluate net, shape = [1, n_features]
        self.eval_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="eval_input")
        # For loss calculation, shape = [1, n_actions], fixed the target
        self.q_target = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name="Q_target")

        with tf.variable_scope("eval_net"):
            ####### Configure of layers #########
            # Collections to store variables, used later when assign to target net
            collections_names = ["eval_net_params", tf.GraphKeys.GLOBAL_VARIABLES]
            # Number of neurons in a layer
            neurons_num = 10
            # Weight initializer
            weight_initializer = tf.random_normal_initializer(0.0, 0.3)
            # Bias initializer
            bias_initializer = tf.constant_initializer(0.1)

            # First layer
            with tf.variable_scope("layer_1"):
                w1 = tf.get_variable(name="w1", shape=[self.n_features, neurons_num], initializer=weight_initializer,
                                     collections=collections_names)
                b1 = tf.get_variable(name="b1", shape=[1, neurons_num], initializer=bias_initializer,
                                     collections=collections_names)
                l1 = tf.nn.relu(tf.matmul(self.eval_input, w1) + b1)  # Activate function is tf.nn.relu

            # Second layer
            with tf.variable_scope("layer_2"):
                w2 = tf.get_variable(name="w2", shape=[neurons_num, self.n_actions], initializer=weight_initializer,
                                     collections=collections_names)
                b2 = tf.get_variable(name="b2", shape=[1, self.n_actions], initializer=bias_initializer,
                                     collections=collections_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        # Definition of loss function
        with tf.variable_scope("loss"):
            # "Fixed Target Network"
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        # Definition of RMSProp optimizer
        with tf.variable_scope("train"):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        # Input for target net, shape = [1, n_features]
        self.target_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="target_input")

        with tf.variable_scope("target_net"):
            # Collections to store variables, used later when assign to target net
            collections_names = ["target_net_params", tf.GraphKeys.GLOBAL_VARIABLES]

            # First layer
            with tf.variable_scope("layer_1"):
                w1 = tf.get_variable(name="w1", shape=[self.n_features, neurons_num], initializer=weight_initializer,
                                     collections=collections_names)
                b1 = tf.get_variable(name="b1", shape=[1, neurons_num], initializer=bias_initializer,
                                     collections=collections_names)
                l1 = tf.nn.relu(tf.matmul(self.target_input, w1) + b1)  # Activate function is tf.nn.relu

            # Second layer
            with tf.variable_scope("layer_2"):
                w2 = tf.get_variable(name="w2", shape=[neurons_num, self.n_actions], initializer=weight_initializer,
                                     collections=collections_names)
                b2 = tf.get_variable(name="b2", shape=[1, self.n_actions], initializer=bias_initializer,
                                     collections=collections_names)
                self.q_next = tf.matmul(l1, w2) + b2

    # Store transition info into memory buffer
    def store_transition(self, state, action, reward, next_state):
        if not hasattr(self, "buffer_counter"):
            self.buffer_counter = 0

        # Construct transition
        transition = np.hstack((state, [action, reward], next_state))
        # Update memory buffer with current transition
        index = self.buffer_counter % self.buffer_size
        self.buffer[index, :] = transition
        # Update buffer counter
        self.buffer_counter += 1

    # Choose action based on observation
    def choose_action(self, observation):
        # Observation, only have batch dimension when feed into tf.placeholder
        observation = observation[np.newaxis, :]

        # _epsilon greedy
        if np.random.uniform() < self._epsilon:
            # Forward feed the observation into evaluate_net and get Q value for every actions
            actions_value = self.session.run(self.q_eval, feed_dict={self.eval_input: observation})
            # Choose an action based on greedy
            action = np.argmax(actions_value)
        else:
            # Randomly choose an action
            action = np.random.randint(0, self.n_actions)
        return action

    # Learn
    def learn(self):
        # Check to update target parameters
        # Update Q target each self.replace_target_iter rounds
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.session.run(self.replace_target_op)
            print("\nQ target net has been updated...\n")

        # Sample batch from memory buffer, the sample_idx is a 1-D list with length of self.batch_size
        if self.buffer_counter > self.buffer_size:
            # Buffer is full
            sample_idx = np.random.choice(self.buffer_size, size=self.batch_size)
        else:
            # Buffer is not full
            sample_idx = np.random.choice(self.buffer_counter, size=self.batch_size)
        # Construct batch buffer
        batch_buffer = self.buffer[sample_idx, :]

        # Note that each transition is in the form of: (state, [action, reward], next_state)

        # Get the action value Q from both target_net and evaluate_net
        q_next, q_eval = self.session.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.target_input: batch_buffer[:, -self.n_features:],    # Next state's features,  last self.n_features cols(!) of samples in batch_buffer
                self.eval_input: batch_buffer[:, :self.n_features],         # Current state's features,  first self.n_features cols(!) of samples in batch_buffer
            }
        )

        # q_target <- q_eval, at this moment q_target remains Q(s_t, a_t)
        # Reason: "First take action A, then we can observe"
        q_target = q_eval.copy()

        batch_idx = np.arange(self.batch_size, dtype=np.int32)
        # Get eval_action
        eval_action_index = batch_buffer[:, self.n_features].astype(int)
        # Get reward
        reward = batch_buffer[:, self.n_features + 1]

        # "Observe"
        # At this moment, q_target has been updated to Q(s_{t+1}, \pi(s_{t+1})) based on greedy policy
        q_target[batch_idx, eval_action_index] = reward + self._gamma * np.max(q_next, axis=1)

        # Train eval_net, make it as close as possible to q_target
        # Since self.loss has got self.q_target, it won't train target_net
        # Since self.loss got self.eval_input, it must train evaluate_net to get self.q_eval to minimize the loss function
        _, self.cost = self.session.run([self.train_op, self.loss],
                                        feed_dict={self.eval_input: batch_buffer[:, :self.n_features],
                                                   self.q_target: q_target
                                                   }
                                        )
        self.cost_history.append(self.cost)

        # Increasing epsilon
        self._epsilon = self._epsilon + self._epsilon_increment if self._epsilon < self._epsilon_max else self._epsilon_max

        self.learn_step_counter += 1

    # Plot cost curve
    def plot_cost_curve(self):
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel("Cost")
        plt.xlabel("Training steps")
        plt.show()



