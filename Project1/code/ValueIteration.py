import numpy as np


# Calculate the value for all actions in a given state, one-step lookahead
# state: state (int)
# V: the value to use as an estimator, vector with the length of env.nS
# Return: max value among these possible values or its arg (depends on whether in output stage)
def calculate_action_value(env, state, V, _discount_factor, output):
    A = np.zeros(env.nA)
    for action in range(env.nA):
        for prob, next_state, reward, isDone in env.P[state][action]:
            A[action] += prob * (reward + _discount_factor * V[next_state])
    if not output:
        return np.max(A)
    else:
        return np.argmax(A)


# env: gridworld
# _theta: Stopping threshold
# _discount_factor: discount factor
# Return: a tuple (policy, V) of the optimal policy and the optimal value function
def value_iteration(env, _theta=0.001, _discount_factor=1.0):
    # Value function
    V = np.zeros(env.nS)
    # Iteration step
    iteration_step = 0
    # policy
    my_policy = np.zeros(env.nS)

    while True:
        # print("Current iteration step: ", iteration_step)
        iteration_step += 1
        # Stop condition
        _delta = 0
        # Update each state
        for state in range(env.nS):
            origin_value = V[state]
            # Calculate best action, one-step lookahead, and update the value function
            V[state] = calculate_action_value(env, state, V, _discount_factor, output=False)
            # Calculate _delta across all states seen so far
            _delta = max(_delta, np.abs(origin_value - V[state]))

        if _delta < _theta:
            print("Stop. Totally", iteration_step, "iterations.")
            break

    # Output a deterministic policy
    for state in range(env.nS):
        my_policy[state] = calculate_action_value(env, state, V, _discount_factor, output=True)

    return my_policy, V
