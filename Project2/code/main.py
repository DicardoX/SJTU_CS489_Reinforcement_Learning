# main.py

import numpy as np
from Gridworld import GridworldEnv
from MonteCarlo import first_visit, every_visit
from TemporalDifference import temporal_difference_learning

# Grid Map Definition
env_for_first_visit = GridworldEnv([6, 6])
env_for_every_visit = GridworldEnv([6, 6])
env_for_temporal_difference = GridworldEnv([6, 6])


def first_visit_policy(env):
    # def first_visit(env, _discount_factor, _theta)
    policy, V = first_visit(env, 0.9, 0.0001)
    print("------------------------------------")
    print("Monte-Carlo First Visit Policy")
    print("------------------------------------")
    print("Reshaped Policy (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("------------------------------------")
    print("Final Value function:")
    print(np.reshape(V, env.shape))
    print("------------------------------------")
    print("")


def every_visit_policy(env):
    # def every_visit(env, _discount_factor, _theta)
    policy, V = every_visit(env, 0.9, 0.0001)
    print("------------------------------------")
    print("Monte-Carlo Every Visit Policy")
    print("------------------------------------")
    print("Reshaped Policy (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("------------------------------------")
    print("Final Value function:")
    print(np.reshape(V, env.shape))
    print("------------------------------------")
    print("")


def temporal_difference_policy(env):
    # def temporal_difference_learning(env, _discount_factor, _alpha, _theta)
    policy, V = temporal_difference_learning(env, 0.9, 0.01, 0.2)
    print("------------------------------------")
    print("Temporal Difference Learning")
    print("------------------------------------")
    print("Reshaped Policy (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("------------------------------------")
    print("Final Value function:")
    print(np.reshape(V, env.shape))
    print("------------------------------------")
    print("")


if __name__ == '__main__':
    # First Visit in Monte-Carlo
    first_visit_policy(env_for_first_visit)
    # Every Visit in Monte-Carlo
    every_visit_policy(env_for_every_visit)
    # Temporal Difference Learning
    temporal_difference_policy(env_for_temporal_difference)






