# QLearning.py

import numpy as np
from Utils import secret_generator, epsilon_greedy_policy
from Gridworld import is_terminal_state, start_state


# Q-Learning Algorithm in Off-policy
def Q_learning(env, _discount_factor, _alpha, _epsilon, basic_rounds):
    print("Begin Q-Learning...")
    # Action value function
    Q = np.zeros([env.nS, env.nA])

    count = 0
    # Repeat loop to update Q function
    while count < basic_rounds:
        # Update counter
        count += 1
        # Display Info
        if count % 10000 == 0:
            print("Iteration: ", count, "/", basic_rounds)
        # Initialize start state
        cur_state = secret_generator.randint(0, 35)

        # Explore the episode (not traverse, since the episode is not deterministic)
        while True:
            # Arrive at the terminal state, break
            if is_terminal_state(cur_state):
                break

            # Choose action based on epsilon-greedy policy
            cur_action = epsilon_greedy_policy(cur_state, env, Q, _epsilon)

            # Take current action
            for prob, next_state, reward, is_done in env.P[cur_state][cur_action]:
                # Choose ASSUMED next action based on NORMAL greedy policy
                assumed_next_action = np.argmax(Q[next_state])
                # Update Q function
                Q[cur_state][cur_action] = Q[cur_state][cur_action] + _alpha * (
                            reward + _discount_factor * Q[next_state][assumed_next_action] - Q[cur_state][cur_action])
                # Move
                cur_state = next_state

    # Policy vector
    my_policy = np.zeros(env.nS)
    # Initialize the policy
    for state in range(env.nS):
        my_policy[state] = -1
    # Generate a deterministic policy
    my_state = start_state
    while not is_terminal_state(my_state):
        # Choose the best action based on greedy
        my_action = np.argmax(Q[my_state])
        # Write into policy
        my_policy[my_state] = my_action
        # Move
        for prob, next_state, reward, is_done in env.P[my_state][my_action]:
            my_state = next_state

    # Return
    return my_policy, Q




