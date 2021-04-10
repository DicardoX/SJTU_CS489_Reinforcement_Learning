# Gridworld.py

import numpy as np
import sys
from gym.envs.toy_text import discrete
from io import StringIO

# Action definition
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Terminal State
terminal_state = 47
# Start State
start_state = 36


# Is terminal state
def is_terminal_state(s):
    return s == terminal_state


# Determine whether at the Cliff
def at_cliff(s):
    return 36 < s < 47


# Gridworld class
class GridworldEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=None):
        # # Initialization for father class
        # super().__init__(nS, nA, P, isd)
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('Shape argument must be a list/tuple of length 2')
        # Declaration
        self.shape = shape

        # Size
        MAX_Y = shape[0]
        MAX_X = shape[1]
        # Total number of grids
        nS = MAX_X * MAX_Y
        # Total number of actions (up, right, down and left)
        nA = 4

        # P
        P = {}
        # Grid container
        grid = np.arange(nS).reshape(shape)
        # Iterator, to visit the grid matrix, 'multi_index' means itr could move to any directions in each iteration
        itr = np.nditer(grid, flags=['multi_index'])

        # Traverse all grids
        while not itr.finished:
            # Number
            s = itr.iterindex
            # Position
            y, x = itr.multi_index
            # Generate options, the number is nA, which means up/right/down/left, P = {0: {0:[], 1:[], 2:[], 3:[]}, 1: ...}
            P[s] = {a: [] for a in range(nA)}

            # Whether it is terminal state
            def is_done(pos):
                return pos == terminal_state
            # Reward settings
            reward = 0.0 if is_done(s) else -1.0
            arrive_cliff_reward = -100                   # If arrive cliff

            # If have reached terminal state
            if is_done(s):
                # [(prob, next_state, reward, done)], prob means transfers probability, all 1.0 in MDP
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                # Calculate the next possible position, if at the edge of the gridworld, just stay
                next_UP = s if y == 0 else s - MAX_X                            # means go up for a gird
                next_RIGHT = s if x == (MAX_X - 1) else s + 1
                next_DOWN = s if y == (MAX_Y - 1) else s + MAX_X
                next_LEFT = s if x == 0 else s - 1
                # If arrive at cliff, the next state will be the start state
                P[s][UP] = [(1.0, next_UP, reward, is_done(next_UP))] if not at_cliff(next_UP) else [(1.0, start_state, arrive_cliff_reward, is_done(start_state))]
                P[s][RIGHT] = [(1.0, next_RIGHT, reward, is_done(next_RIGHT))] if not at_cliff(next_RIGHT) else [(1.0, start_state, arrive_cliff_reward, is_done(start_state))]
                P[s][DOWN] = [(1.0, next_DOWN, reward, is_done(next_DOWN))] if not at_cliff(next_DOWN) else [(1.0, start_state, arrive_cliff_reward, is_done(start_state))]
                P[s][LEFT] = [(1.0, next_LEFT, reward, is_done(next_LEFT))] if not at_cliff(next_LEFT) else [(1.0, start_state, arrive_cliff_reward, is_done(start_state))]

            # Update iterator
            itr.iternext()

        # Initial state distribution is uniform
        initial_state_distribution = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        # Save parameters
        super(GridworldEnv, self).__init__(nS, nA, P, initial_state_distribution)

    # Overwrite render function
    # def _render(self, mode='human', close=False):
    #     # Close
    #     if close:
    #         return
    #     # sys.stdout: the input control port of python, print == sys.stdout.write()
    #     # StringIO(): read/write str in memory
    #     outfile = StringIO() if mode == 'ansi' else sys.stdout
    #
    #     # Initialize map
    #     grid = np.arange(self.nS).reshape(self.shape)
    #     # Iterator
    #     itr = np.nditer(grid, flags=['multi_index'])
    #
    #     # Traverse all grids
    #     while not itr.finished:
    #         s = itr.iterindex
    #         y, x = itr.multi_index
    #         # terminal state
    #         if self.s == s:
    #             output = " x "
    #         elif s == terminal_state_1 or s == terminal_state_2:
    #             output = " T "
    #         else:
    #             # Normal grid
    #             output = " o "
    #
    #         # If it is a new line
    #         if x == 0:
    #             # Remove the idle char on the left side of output
    #             output = output.lstrip()
    #         if x == self.shape[1] - 1:
    #             # Remove the idle char on the right side of output
    #             output = output.rstrip()
    #
    #         outfile.write(output)
    #
    #         if x == self.shape[1] - 1:
    #             outfile.write("\n")
    #
    #         itr.iternext()



