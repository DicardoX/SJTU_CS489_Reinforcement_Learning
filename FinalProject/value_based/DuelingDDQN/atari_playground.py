# atari_playground.py

import numpy as np
import gym
import torch
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from RL_brain import Dueling_Double_DeepQNetwork
from atari_wrappers import wrap_deepmind, make_atari
from utils import save_log_loss, save_log_score, save_model_params, date_in_string

# Start point of learning, for store transitions into buffer
LEARN_START = 50000
# Total epochs
TOTAL_STEPS = 10000000
# Target net update iter
TARGET_NET_UPDATE_ITER = 10000
# Evaluate iter
EVALUATE_ITER = 20

# Arguments and parser
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="BreakoutNoFrameskip-v4", help="Name & version of atari game")
args = parser.parse_args()

# Instantiation of Dueling_DQN
Dueling_DDQN_model = Dueling_Double_DeepQNetwork(args.env_name)

max_r = 0
mean_r = 0
loss = 0
avg_q = 0
running_loss = 0
max_mean_r = 0
sum_r = 0
done = True
i = 0


# Evaluate the performance
def evaluate(step, eva_net, env, num_episode=15):
    env = wrap_deepmind(env)
    e_rewards = []
    for i in range(num_episode):
        img = env.reset()
        sum_r = 0
        done = False
        state_buffer = []
        for i in range(5):
            state_buffer.append(img)
        s = state_buffer[1:5]
        while not done:
            a = Dueling_DDQN_model.choose_action(s, is_train=False)

            img, r, done, info = env.step(a)
            sum_r += r
            state_buffer.pop(0)
            state_buffer.append(img)
            s_ = state_buffer[1:5]
            s = s_

        e_rewards.append(sum_r)

    f = open("./log/reward.txt", 'a')
    f.write("%f, %d, %d\n" % (float(sum(e_rewards)) / float(num_episode), step, num_episode))
    f.close()


progressive = tqdm(range(TOTAL_STEPS), total=TOTAL_STEPS, ncols=50, leave=False, unit='b')
for step in progressive:

    if done and Dueling_DDQN_model.env.was_real_done is True:
        mean_r += sum_r
        if sum_r > max_r:
            max_r = sum_r
        #
        if (i + 1) % EVALUATE_ITER == 0:
            save_log_score(i, mean_r / EVALUATE_ITER, max_r)
            print('    ', i, mean_r / EVALUATE_ITER, max_r)
            if mean_r > max_mean_r:
                max_mean_r = mean_r
                save_model_params(Dueling_DDQN_model.eval_net)
            max_r, mean_r = 0, 0

        sum_r = 0
        i += 1

    if done:
        s = Dueling_DDQN_model.initialize_state()
        img, _, _, _ = Dueling_DDQN_model.env.step(1)

    a = Dueling_DDQN_model.choose_action(s)

    img, r, done, info = Dueling_DDQN_model.env.step(a)
    sum_r += r
    Dueling_DDQN_model.state_buffer.pop(0)
    Dueling_DDQN_model.state_buffer.append(img)

    s_ = Dueling_DDQN_model.state_buffer[1:5]
    Dueling_DDQN_model.store_transition(s, a, r, s_, done)
    s = s_

    if len(Dueling_DDQN_model.buffer) > LEARN_START and Dueling_DDQN_model.step % 4:
        loss, avg_q = Dueling_DDQN_model.learn()
    running_loss += loss
    if Dueling_DDQN_model.step % 500 == 0:
        running_loss /= 250
    if Dueling_DDQN_model.step % TARGET_NET_UPDATE_ITER == 0:
        save_log_loss(step, loss, avg_q)
        Dueling_DDQN_model.target_net.load_state_dict(Dueling_DDQN_model.eval_net.state_dict())
    if Dueling_DDQN_model.step % 500 == 0:
        running_loss = 0
    if (Dueling_DDQN_model.step + 1) % 50000 == 0:
        evaluate(step, Dueling_DDQN_model.eval_net, Dueling_DDQN_model.unwrapped_env, num_episode=10)
