# main.py

import gym
import numpy as np

# Need to pip install pyglet==1.15.1


if __name__ == '__main__':
    # env = gym.make('MountainCar-v0')
    # env.reset()
    # for _ in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample())  # take a random action
    # env.close()

    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(arr[0, :])
    # Output: [1, 2, 3]
    print(arr[:, 1])
    # Output: [2, 5]
    print(arr[:, :1])
    # Output: [[1]
    #           [4]]
    print(arr[:, -2])
    # Output: [2, 5]
    print(arr[:, -2:])
    # Output: [[2 3]
    #           [5 6]]

