# Utils.py

import numpy as np
import secrets

# Define secrets generator
secret_generator = secrets.SystemRandom()


# Calculate best action based on Q
def calculate_action_value_based_on_Q(state, Q):
    return np.argmax(Q[state])


# Implementation of epsilon-greedy policy
# _epsilon: 0.1 by default
# Return: a action decided by epsilon-greedy policy
def epsilon_greedy_policy(state, env, Q, _epsilon=0.1):
    # m in epsilon-greedy policy
    m_size = env.nA
    # Best action
    best_action = calculate_action_value_based_on_Q(state, Q)
    # If epsilon is 0
    if _epsilon == 0:
        return best_action

    # Construct probability dict = {action1: prob1, action2: prob2, ...}
    prob_dict = {}
    cur_prob_sum = 0
    for action in range(env.nA):
        # In case of accuracy loss, which means that cur_prob_sum is not 1 in the end
        if action == env.nA - 1:
            # Last action
            prob_dict[action] = 1.0
            cur_prob_sum = 1.0
            break

        if action == best_action:
            prob_dict[action] = cur_prob_sum + _epsilon / m_size + 1 - _epsilon
            cur_prob_sum += _epsilon / m_size + 1 - _epsilon
        else:
            prob_dict[action] = cur_prob_sum + _epsilon / m_size
            cur_prob_sum += _epsilon / m_size

    # Determine which action to do by epsilon-greedy policy
    # Random seed
    random_seed = secret_generator.randint(0, 1000000) * 0.000001
    # print("Random seed: ", random_seed, " prob dict: ", prob_dict)
    # Check
    for action in prob_dict:
        if random_seed <= prob_dict[action]:
            return action
