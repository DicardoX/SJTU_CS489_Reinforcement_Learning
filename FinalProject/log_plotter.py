# log_plotter.py

from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

# Filter size for medium filter
filter_size = 13
# Whether smooth or not
is_smooth = False


# Read log loss and Q value from txt
def read_log_loss_and_Q_value_txt(file_path):
    # Loss curve
    loss_list = []
    # Q value curve
    q_value_list = []

    with open(file_path, "r") as f:
        data = f.readlines()
        for i in range(len(data)):
            message = data[i].split(" ")[0]
            message = message.replace("\n", "")
            message = message.split("\t")

            # Loss
            cur_loss = float(message[2])
            loss_list.append(cur_loss)
            # Q value
            cur_q_value = float(message[3])
            q_value_list.append(cur_q_value)
    return loss_list, q_value_list


# Read log score from txt
def read_log_score_txt(file_path):
    # Average score curve
    avg_score_list = []
    # Local max curve
    max_score_list = []

    with open(file_path, "r") as f:
        data = f.readlines()
        for i in range(len(data)):
            message = data[i].split(" ")[0]
            message = message.replace("\n", "")
            message = message.split("\t")

            # Avg
            cur_avg_score = float(message[2])
            avg_score_list.append(cur_avg_score)
            # Max
            cur_max_score = float(message[3])
            max_score_list.append(cur_max_score)
    return avg_score_list, max_score_list


# Plot
def plotter():
    # Read txt
    # Dueling DQN
    loss_list_1, q_value_list_1 = read_log_loss_and_Q_value_txt("./value_based/DuelingDQN/log/log_loss.txt")
    avg_score_list_1, max_score_list_1 = read_log_score_txt("./value_based/DuelingDQN/log/log_score.txt")
    # Dueling DDQN
    loss_list_2, q_value_list_2 = read_log_loss_and_Q_value_txt("./value_based/DuelingDDQN/log/log_loss.txt")
    avg_score_list_2, max_score_list_2 = read_log_score_txt("./value_based/DuelingDDQN/log/log_score.txt")

    # Smooth
    if is_smooth:
        avg_score_list_1 = signal.medfilt(avg_score_list_1, kernel_size=filter_size)
        avg_score_list_2 = signal.medfilt(avg_score_list_2, kernel_size=filter_size)
        max_score_list_1 = signal.medfilt(max_score_list_1, kernel_size=filter_size)
        max_score_list_2 = signal.medfilt(max_score_list_2, kernel_size=filter_size)
        loss_list_1 = signal.medfilt(loss_list_1, kernel_size=filter_size)
        loss_list_2 = signal.medfilt(loss_list_2, kernel_size=filter_size)
        q_value_list_1 = signal.medfilt(q_value_list_1, kernel_size=filter_size)
        q_value_list_2 = signal.medfilt(q_value_list_2, kernel_size=filter_size)

    # Avg score comparison
    avg_figure, avg_ax = plt.subplots()
    x_range = np.arange(0, min(len(avg_score_list_1), len(avg_score_list_2)))
    avg_ax.plot(x_range, avg_score_list_1[0:len(x_range)], label="DuelingDQN")
    avg_ax.plot(x_range, avg_score_list_2[0:len(x_range)], label="DuelingDDQN")
    avg_ax.set_xlabel("Episode")
    avg_ax.set_ylabel("Score")
    smooth_msg = "Smooth" if is_smooth else ""
    avg_ax.set_title(smooth_msg + " Average Score Curve in BreakoutNoFrameskip-v4")
    avg_ax.legend()
    if is_smooth:
        save_path = "./output/value_based/" + smooth_msg.casefold() + "_avg_score.png"
    else:
        save_path = "./output/value_based/avg_score.png"
    plt.savefig(save_path)
    plt.show()

    # Max score comparison
    avg_figure, avg_ax = plt.subplots()
    x_range = np.arange(0, min(len(max_score_list_1), len(max_score_list_2)))
    avg_ax.plot(x_range, max_score_list_1[0:len(x_range)], label="DuelingDQN")
    avg_ax.plot(x_range, max_score_list_2[0:len(x_range)], label="DuelingDDQN")
    avg_ax.set_xlabel("Episode")
    avg_ax.set_ylabel("Score")
    smooth_msg = "Smooth" if is_smooth else ""
    avg_ax.set_title(smooth_msg + " Max Score Curve in BreakoutNoFrameskip-v4")
    avg_ax.legend()
    if is_smooth:
        save_path = "./output/value_based/" + smooth_msg.casefold() + "_max_score.png"
    else:
        save_path = "./output/value_based/max_score.png"
    plt.savefig(save_path)
    plt.show()

    # Loss comparison
    avg_figure, avg_ax = plt.subplots()
    x_range = np.arange(0, min(len(loss_list_1), len(loss_list_2)))
    avg_ax.plot(x_range, loss_list_1[0:len(x_range)], label="DuelingDQN")
    avg_ax.plot(x_range, loss_list_2[0:len(x_range)], label="DuelingDDQN")
    avg_ax.set_xlabel("Episode")
    avg_ax.set_ylabel("Loss")
    smooth_msg = "Smooth" if is_smooth else ""
    avg_ax.set_title(smooth_msg + " Loss Curve in BreakoutNoFrameskip-v4")
    avg_ax.legend()
    if is_smooth:
        save_path = "./output/value_based/" + smooth_msg.casefold() + "_loss.png"
    else:
        save_path = "./output/value_based/loss.png"
    plt.savefig(save_path)
    plt.show()

    # Q value comparison
    avg_figure, avg_ax = plt.subplots()
    x_range = np.arange(0, min(len(q_value_list_1), len(q_value_list_2)))
    avg_ax.plot(x_range, q_value_list_1[0:len(x_range)], label="DuelingDQN")
    avg_ax.plot(x_range, q_value_list_2[0:len(x_range)], label="DuelingDDQN")
    avg_ax.set_xlabel("Episode")
    avg_ax.set_ylabel("Q Value")
    smooth_msg = "Smooth" if is_smooth else ""
    avg_ax.set_title(smooth_msg + " Q Value Curve in BreakoutNoFrameskip-v4")
    avg_ax.legend()
    if is_smooth:
        save_path = "./output/value_based/" + smooth_msg.casefold() + "_q_value.png"
    else:
        save_path = "./output/value_based/q_value.png"
    plt.savefig(save_path)
    plt.show()


# Read npy
def read_npy(file_path):
    npy_file = np.load(file_path)
    return npy_file


# Plot for TD3
def td3_plotter():
    # Read npy
    td3_list = read_npy("./policy_based/TD3/results/TD3_Humanoid-v2_0.npy")
    ddpg_list = read_npy("./policy_based/TD3/results/DDPG_Humanoid-v2_0.npy")

    # Average Reward Evaluate curve
    avg_figure, avg_ax = plt.subplots()
    x_range = np.arange(0, min(len(td3_list), len(ddpg_list)))
    avg_ax.plot(x_range, td3_list[0:len(x_range)], label="TD3")
    avg_ax.plot(x_range, ddpg_list[0:len(x_range)], label="DDPG")
    avg_ax.set_xlabel("Episode")
    avg_ax.set_ylabel("Score")
    avg_ax.set_title("Average Evaluate Reward in Humanoid-v2")
    avg_ax.legend()
    save_path = "./output/policy_based/avg_eval_reward.png"
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # Atari plotter
    # plotter()

    # TD3 plotter
    td3_plotter()
