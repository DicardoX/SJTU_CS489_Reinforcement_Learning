# MonteCarlo.py

import numpy as np
from Utils import generate_random_path, calculate_action_value, get_reward

# At least iterate 30000 rounds, in order to guarantee the stability of the algorithm
basic_rounds = 30000


# First-Visit in Monte-Carlo
def first_visit(env, _discount_factor, _theta):
    print("Begin First-visit in Monte-Carlo Learning...")
    # Value function
    V = np.zeros(env.nS)
    # Policy vector
    my_policy = np.zeros([env.nS, env.nA])
    # Initialize the policy
    for state in range(env.nS):
        for action in range(env.nA):
            my_policy[state][action] = 0.25
    # Number vector
    N = np.zeros(env.nS)
    # Return vector
    S = np.zeros(env.nS)

    _delta = 1
    satisfied_count = 0
    count = 0
    # Repeat loop to update value function
    while _delta >= _theta or satisfied_count < 1 or count < basic_rounds:
        count += 1
        if count % 10000 == 0:
            print("Iteration: ", count, "/", basic_rounds)
        # Need two satisfied times to quit, make this algorithm more stable
        if _delta < _theta:
            satisfied_count += 1

        # Tmp S, keep state to implement first-visit
        tmp_S = S.copy()
        # Tmp N
        tmp_N = N.copy()
        # Generate random path based on current policy
        random_path = generate_random_path(env, my_policy)
        _delta = 0
        G = 0
        # Traverse the path
        for t in range(len(random_path) - 1, -1, -1):
            G = _discount_factor * G + get_reward(random_path[t])
            tmp_S[random_path[t]] = S[random_path[t]] + G
            tmp_N[random_path[t]] = N[random_path[t]] + 1
        # Update return vector S, value function V and number vector N
        for state in range(env.nS):
            N[state] = tmp_N[state]
            S[state] = tmp_S[state]
            if N[state] > 0:
                _delta = max(_delta, abs(V[state] - S[state] / N[state]))
                V[state] = S[state] / N[state]

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
    # print(count)
    return my_policy, V


# Every-Visit in Monte-Carlo
def every_visit(env, _discount_factor, _theta):
    print("Begin Every-visit in Monte-Carlo Learning...")
    # Value function
    V = np.zeros(env.nS)
    # Policy vector
    my_policy = np.zeros([env.nS, env.nA])
    # Initialize the policy
    for state in range(env.nS):
        for action in range(env.nA):
            my_policy[state][action] = 0.25
    # Number vector
    N = np.zeros(env.nS)
    # Return vector
    S = np.zeros(env.nS)

    _delta = 1
    satisfied_count = 0
    count = 0
    # Repeat loop to update value function
    while _delta >= _theta or satisfied_count < 1 or count < basic_rounds:
        count += 1
        if count % 10000 == 0:
            print("Iteration: ", count, "/", basic_rounds)
        # Need two satisfied times to quit, make this algorithm more stable
        if _delta < _theta:
            satisfied_count += 1

        # Generate random path
        random_path = generate_random_path(env, my_policy)
        _delta = 0
        G = 0
        # Traverse the path and update value function V
        for t in range(len(random_path) - 1, -1, -1):
            G = _discount_factor * G + get_reward(random_path[t])
            S[random_path[t]] += G
            N[random_path[t]] += 1
            _delta = max(_delta, abs(V[random_path[t]] - S[random_path[t]] / N[random_path[t]]))
            V[random_path[t]] = S[random_path[t]] / N[random_path[t]]

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
    # print(count)
    return my_policy, V






