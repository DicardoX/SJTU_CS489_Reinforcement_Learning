# MountainCar_v0.py

import gym
import numpy as np
from matplotlib import pyplot as plt
from RL_brain import DeepQNetwork, Double_DeepQNetwork
import time
import os

# Environment
env = gym.make("MountainCar-v0")
# Remove the limits(e.g. step limits) in environment encapsulation
env = env.unwrapped

# Instantiation of DQN
DQN_model = DeepQNetwork(n_actions=3, n_features=2, neurons_num=16, learning_rate=0.0001, epsilon_greedy=0.9,
                         replace_target_iter=300,
                         buffer_size=4096, batch_size=64, epsilon_greedy_increment=0.0001)
# Instantiation of Double_DQN
Double_DQN_model = Double_DeepQNetwork(n_actions=3, n_features=2, neurons_num=16, learning_rate=0.0001,
                                       epsilon_greedy=0.9,
                                       replace_target_iter=300,
                                       buffer_size=4096, batch_size=64, epsilon_greedy_increment=0.0001)


# Step list
my_step_list = []
# Episode cost list
my_episode_cost_list = []
# Episode time list
my_episode_time_list = []
# Message list
message_list = []


# Plot
def plot_info():
    output_path = "./output/"

    # Plot step list
    step_list_x = np.arange(len(my_step_list[0]))
    plt.plot(step_list_x, my_step_list[0], color="tab:blue")
    plt.plot(step_list_x, my_step_list[1], color="tab:orange")
    plt.xlabel("Epochs")
    plt.ylabel("Step of episode")
    plt.legend(["DQN", "Double DQN"])
    plt.savefig(output_path + "step_list")
    plt.show()

    # Plot episode cost list
    episode_cost_list_x = np.arange(len(my_episode_cost_list[0]))
    plt.plot(episode_cost_list_x, my_episode_cost_list[0], color="tab:blue")
    plt.plot(episode_cost_list_x, my_episode_cost_list[1], color="tab:orange")
    plt.xlabel("Epochs")
    plt.ylabel("Episode cost")
    plt.legend(["DQN", "Double DQN"])
    plt.savefig(output_path + "episode_cost_list")
    plt.show()

    # Plot episode time list
    episode_time_list_x = np.arange(len(my_episode_time_list[0]))
    plt.plot(episode_time_list_x, my_episode_time_list[0], color="tab:blue")
    plt.plot(episode_time_list_x, my_episode_time_list[1], color="tab:orange")
    plt.xlabel("Epochs")
    plt.ylabel("Episode time")
    plt.legend(["DQN", "Double DQN"])
    plt.savefig(output_path + "episode_time_list")
    plt.show()

    # Plot step cost
    DQN_model.plot_cost_curve()
    Double_DQN_model.plot_cost_curve()


def main(choice_index):
    # Step list
    step_list = []
    # Episode cost list
    episode_cost_list = []
    # Episode time list
    episode_time_list = []

    # Write path
    write_path = "./output/console_log.txt" if choice_index == 0 else "./output/console_log_for_DDQN.txt"

    # Step counter
    steps_counter = 0

    print("--------------------------------------------")
    print("Begin", "DQN" if choice_index == 0 else "Double DQN", "training...")
    print("--------------------------------------------")

    for i in range(total_episodes):
        # Observation
        observation = env.reset()
        # Total reward in an episode
        episode_reward = 0
        tmp_steps = 0

        # Clear self.episode_cost
        if choice_index == 0:
            DQN_model.episode_cost = 0
        else:
            Double_DQN_model.episode_cost = 0

        # Time mark
        time_mark = time.time()

        while True:
            # Render the environment
            env.render()
            # Choose action
            action = DQN_model.choose_action(
                observation=observation) if choice_index == 0 else Double_DQN_model.choose_action(
                observation=observation)
            # Get feedback from environment
            new_observation, reward, done, info = env.step(action)
            # Parse the observation
            position, velocity = new_observation
            # Reward policy: the higher, the better
            reward = abs(position - (-0.5))

            # Store the transition
            if choice_index == 0:
                DQN_model.store_transition(observation, action, reward, new_observation)
            else:
                Double_DQN_model.store_transition(observation, action, reward, new_observation)

            # Learn
            if steps_counter > 1000:
                if choice_index == 0:
                    DQN_model.learn()
                else:
                    Double_DQN_model.learn()
            # Update reward
            episode_reward += reward

            # If done
            if done:
                sub_message = "| Get" if new_observation[0] >= env.unwrapped.goal_position else "| ----"
                message = "Episode: " + str(i) + " " + sub_message + " | Episode reward: " + str(
                    round(episode_reward, 3)) + \
                          " | Episode steps: " + str(tmp_steps) + " | Episode cost: " + str(
                    round(DQN_model.episode_cost if choice_index == 0 else Double_DQN_model.episode_cost, 3)) + \
                          " | Episode time: " + str(round(time.time() - time_mark, 3)) + " seconds"
                print(message)
                # Update message list
                message_list.append(message)
                break

            # If not done
            # Transform state
            observation = new_observation
            steps_counter += 1
            tmp_steps += 1

        # Update steps list
        step_list.append(tmp_steps)
        # Update episode cost list
        episode_cost_list.append(DQN_model.episode_cost if choice_index == 0 else Double_DQN_model.episode_cost)
        # Update episode time list
        episode_time_list.append(round(time.time() - time_mark))

    # Write message list
    # File operation
    if os.path.exists(write_path):
        os.remove(write_path)
    f = open(write_path, "w")
    for i in range(len(message_list)):
        f.writelines(message_list[i])
        f.write("\n")
    f.close()

    my_step_list.append(step_list)
    my_episode_cost_list.append(episode_cost_list)
    my_episode_time_list.append(episode_time_list)


if __name__ == '__main__':
    # Total episodes
    total_episodes = 300

    # DQN
    # Choose use DQN or DDQN: 0 for DQN, 1 for DDQN
    choice_index = 0
    main(choice_index)

    # Double DQN
    choice_index = 1
    main(choice_index)

    # Plot
    plot_info()
