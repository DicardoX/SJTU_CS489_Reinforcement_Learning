import numpy as np
import gym

# Action definition
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


# Gridworld class
class GridworldEnv:
    def __init__(self, shape=None):
        # Whether shape argument is correct
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('Shape argument must be a list/tuple of length 2')

        self.shape = shape
