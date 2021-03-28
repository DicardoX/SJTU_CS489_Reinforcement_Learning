# MonteCarlo.py

import numpy as np
import secrets
from Gridworld import is_terminal_state, terminal_state_1, terminal_state_2, UP, RIGHT, DOWN, LEFT
from Utils import generate_random_path, get_reward, calculate_action_value

# Define secrets generator
secret_generator = secrets.SystemRandom()

# At least iterate 50000 rounds, in order to guarantee the stability of the algorithm
basic_rounds = 50000


# Temporal Difference Learning
# _alpha: step size
def temporal_difference_learning(env, _discount_factor, _alpha, _theta):
    print("Begin Temporal Difference Learning...")
    # Value function
    V = np.zeros(env.nS)
    # Policy vector
    my_policy = np.zeros([env.nS, env.nA])
    # Initialize the policy
    for state in range(env.nS):
        for action in range(env.nA):
            my_policy[state][action] = 0.25

    _delta = 1
    count = 0
    # Repeat loop to update value function
    while _delta >= _theta or count < basic_rounds:
        count += 1
        if count % 10000 == 0:
            print("Iteration: ", count, "/", basic_rounds)

        # Get random path
        random_path = generate_random_path(env, my_policy)
        _delta = 0
        # Traverse the path until S is the terminal
        for t in range(0, len(random_path) - 1, 1):
            # Update V
            ori_V = V[random_path[t]]
            V[random_path[t]] = V[random_path[t]] + _alpha * (get_reward(random_path[t]) + _discount_factor * V[random_path[t + 1]] - V[random_path[t]])
            _delta = max(_delta, abs(ori_V - V[random_path[t]]))

    # Output a deterministic policy
    for state in range(env.nS):
        # Get optimal direction
        direction = calculate_action_value(env, state, V, _discount_factor)
        # Update policy (make choice)
        for action in range(env.nA):
            if action == direction:
                my_policy[state][action] = 1
            else:
                my_policy[state][action] = 0
    print(count)
    return my_policy, V
