# main.py

import numpy as np
from Gridworld import GridworldEnv
from Sarsa import Sarsa_learning
from QLearning import Q_learning
from Arguments import _discount_factor, _alpha, _epsilon, basic_rounds

# Grid Map Definition
# Order: from top left is 0, to right is 1, 2, ..., to down is 12, 24, ...
env_for_Sarsa = GridworldEnv([4, 12])
env_for_QLearning = GridworldEnv([4, 12])


def Sarsa_learning_policy(env):
    # Sarsa learning
    policy, Q = Sarsa_learning(env, _discount_factor, _alpha, _epsilon, basic_rounds)
    # Output
    print("------------------------------------")
    print("Sarsa Learning Policy")
    print("------------------------------------")
    print("Reshaped Policy (-1=not_visited (or end for terminal state), 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT):")
    print(np.reshape(policy, env.shape))
    print("------------------------------------")
    print("Final Action Value Function (only show the max action value in each state):")
    print(np.reshape(np.max(Q, axis=1), env.shape))
    print("------------------------------------")
    print("")


def Q_learning_policy(env):
    # Q-learning
    policy, Q = Q_learning(env, _discount_factor, _alpha, _epsilon, basic_rounds)
    # Output
    print("------------------------------------")
    print("Q-Learning Policy")
    print("------------------------------------")
    print("Reshaped Policy (-1=not_visited (or end for terminal state), 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT):")
    print(np.reshape(policy, env.shape))
    print("------------------------------------")
    print("Final Action Value Function (only show the max action value in each state):")
    print(np.reshape(np.max(Q, axis=1), env.shape))
    print("------------------------------------")
    print("")


if __name__ == '__main__':
    # Sarsa Learning
    Sarsa_learning_policy(env_for_Sarsa)
    # Q Learning
    Q_learning_policy(env_for_QLearning)
