# MonteCarlo.py

import numpy as np
import secrets
from Gridworld import is_terminal_state, terminal_state_1, terminal_state_2, UP, RIGHT, DOWN, LEFT

# Define secrets generator
secret_generator = secrets.SystemRandom()


# Calculate the value for all actions in a given state, one-step lookahead
# state: state (int)
# V: value function, vector with the length of env.nS
# _discount_factor: discount factor
# Return: max value arg among these possible values
def calculate_action_value(env, state, V, _discount_factor):
    A = np.zeros(env.nA)
    for action in range(env.nA):
        for prob, next_state, reward, isDone in env.P[state][action]:
            A[action] += prob * (reward + _discount_factor * V[next_state])
    return np.argmax(A)


# Generate random path based on \pi
# Random policy: choice a random number between 0 ~ 1, form the possibility of four directions into a set of intervals.
# The interval order is: UP - RIGHT - DOWN - LEFT, and if this random number is in UP interval, we choose UP in this policy
# episode.
def generate_random_path(env, my_policy):
    # Path
    path = []
    # Random initial state
    initial_state = terminal_state_1
    while is_terminal_state(initial_state):
        initial_state = secret_generator.randint(0, 35)

    current_state = initial_state
    while not is_terminal_state(current_state):
        path.append(current_state)
        # Random seed
        rand = secret_generator.randint(0, 1000) * 0.001
        if rand < my_policy[current_state][UP]:
            for prob, next_state, reward, isDone in env.P[current_state][UP]:
                current_state = next_state
                continue
        elif rand < my_policy[current_state][UP] + my_policy[current_state][RIGHT]:
            for prob, next_state, reward, isDone in env.P[current_state][RIGHT]:
                current_state = next_state
                continue
        elif rand < my_policy[current_state][UP] + my_policy[current_state][RIGHT] + my_policy[current_state][DOWN]:
            for prob, next_state, reward, isDone in env.P[current_state][DOWN]:
                current_state = next_state
                continue
        else:
            for prob, next_state, reward, isDone in env.P[current_state][LEFT]:
                current_state = next_state
                continue
    path.append(current_state)

    # Limit the length of the path
    if len(path) > 30:
        return generate_random_path(env, my_policy)
    else:
        return path


# Reward
def get_reward(pos):
    if pos == terminal_state_1 or pos == terminal_state_2:
        return 0
    return -1
