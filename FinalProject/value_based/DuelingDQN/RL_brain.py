# RL_brain.py

import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.nn.functional import relu, mse_loss
import numpy as np
import cv2
import gym
import random
from collections import namedtuple

from atari_wrappers import wrap_deepmind, make_atari

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'is_done'))

# Buffer size
BUFFER_SIZE = 1000000
# Size of state space, 4 + 1
N_STATES = 5
# Epsilon greedy coefficient
EPSILON_START = 1
EPSILON_END = 0.1
# Eps for optimizer
EPS = 1.5e-4
# Learning rate
LR = 0.0000625
# LR = 0.00025
# Batch size
BATCH_SIZE = 32
# Discount factor
GAMMA = 0.99
# When epsilon begins to decrease
LEARN_START = 50000

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNet(nn.Module):
    def __init__(self, n_actions, seed):
        super(QNet, self).__init__()
        # Random seed
        self.seed = torch.manual_seed(seed)
        # Size of action space
        self.n_actions = n_actions
        # --------------------- Network layers ---------------------
        # Conv layer 1
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        # Conv layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # Conv layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # --------------------- Value Net --------------------------
        # Analyse the value of this state
        # Fully connect layer 1
        self.value_fc1 = nn.Linear(3136, 512)
        # Fully connect layer 2
        self.value_fc2 = nn.Linear(512, 1)

        # ------------------- Advantage Net ------------------------
        # Analyse the advantage of each action
        # Fully connect layer 1
        self.advantage_fc1 = nn.Linear(3136, 512)
        # Fully connect layer 2
        self.advantage_fc2 = nn.Linear(512, n_actions)

    def forward(self, state):
        # Batch size
        batch_size = state.size(0)
        # ------------------ Network connections -------------------
        state = state / 255.
        conv1_res = relu(self.conv1(state))
        conv2_res = relu(self.conv2(conv1_res))
        conv3_res = relu(self.conv3(conv2_res))
        conv3_res = conv3_res.view(batch_size, -1)
        # Value part
        value_fc1_res = relu(self.value_fc1(conv3_res))
        value_fc2_res = self.value_fc2(value_fc1_res).expand(batch_size, self.n_actions)
        # Advantage part
        advantage_fc1_res = relu(self.advantage_fc1(conv3_res))
        advantage_fc2_res = self.advantage_fc2(advantage_fc1_res)
        # Output
        out = value_fc2_res + (
                advantage_fc2_res - advantage_fc2_res.mean(1).unsqueeze(1).expand(batch_size, self.n_actions))
        return out

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class Dueling_DeepQNetwork:
    def __init__(self, env_name):
        # Make atari env (unwrapped)
        self.unwrapped_env = make_atari(env_name)
        # Wrapped env
        self.env = wrap_deepmind(self.unwrapped_env, frame_stack=False, episode_life=True, clip_rewards=True)
        # Size of action space
        self.n_actions = self.env.action_space.n
        # Size of state space
        self.n_states = N_STATES
        # Total steps
        self.step = 0
        # Buffer
        self.buffer = []
        # State buffer
        self.state_buffer = []
        # Seed
        self.seed = 1
        # Epsilon greedy coefficient
        self.epsilon = EPSILON_START

        # ------------------- Evaluate Net ----------------------
        self.eval_net = QNet(self.n_actions, self.seed).to(device)
        # Initialize the weights of eval net
        self.eval_net.apply(self.eval_net.init_weights)
        # -------------------- Target Net -----------------------
        self.target_net = QNet(self.n_actions, self.seed).to(device)
        # Load the model of eval net
        self.target_net.load_state_dict(self.eval_net.state_dict())
        # Do not let target net to train temporarily
        self.target_net.eval()

        # Optimizer, based on the parameters in eval net
        self.optimizer = optimizer.Adam(self.eval_net.parameters(), lr=LR, eps=EPS)

    # Initialize the state
    def initialize_state(self):
        # Image
        img = self.env.reset()
        for i in range(self.n_states):
            self.state_buffer.append(img)
        return self.state_buffer[1:5]

    # Choose action
    def choose_action(self, state, is_train=True):
        # Determine epsilon
        if is_train:
            if len(self.buffer) >= LEARN_START:
                self.epsilon -= (EPSILON_START - EPSILON_END) / BUFFER_SIZE
                self.epsilon = max(self.epsilon, EPSILON_END)
            epsilon = self.epsilon
        else:
            epsilon = 0.05

        # Epsilon Greedy Algorithm
        if np.random.uniform() > epsilon:
            state = torch.unsqueeze(torch.tensor(np.array(state, dtype=np.float32), device=device, dtype=torch.float32), 0).to(
                device)
            # Get Q value from eval net
            q_value = self.eval_net(state).detach()
            # Choose action based on Q value by using Greedy Algorithm
            action = torch.argmax(q_value).item()
        else:
            # Randomly sample the action
            action = self.env.action_space.sample()
        return action

    # Store transition (experience)
    def store_transition(self, state, action, reward, next_state, is_done):
        # Update step
        self.step += 1
        # Construct transition
        transition = [state, action, reward, next_state, is_done]
        # Throw oldest transition
        if len(self.buffer) >= BUFFER_SIZE:
            self.buffer.pop(0)
        # Store
        self.buffer.append(transition)

    # Model learn
    def learn(self):
        # Sample from buffer with the size of BATCH_SIZE
        sample = random.sample(self.buffer, BATCH_SIZE)
        # Form batch
        batch = Experience(*zip(*sample))

        batch_state = torch.tensor(np.array(batch.state, dtype=np.float32), device=device, dtype=torch.float32).to(
            device)
        batch_action = torch.tensor(batch.action, device=device).unsqueeze(1).to(device)
        batch_reward = torch.tensor(np.array(batch.reward, dtype=np.float32), device=device,
                                    dtype=torch.float32).unsqueeze(1).to(
            device)
        batch_next_state = torch.tensor(np.array(batch.next_state, dtype=np.float32), device=device,
                                        dtype=torch.float32).to(device)
        batch_is_done = torch.tensor(np.array(batch.is_done, dtype=np.float32), device=device,
                                     dtype=torch.float32).unsqueeze(1).to(
            device)
        # Q value in eval net
        q_eval = torch.gather(self.eval_net(batch_state), 1, batch_action)
        # Average Q value in this batch
        avg_q = torch.sum(q_eval.detach()) / BATCH_SIZE
        q_eval = q_eval.to(device)

        # Double DQN
        # argmax = self.eval_net(batch_next_state).detach().max(1)[1].long()

        # q_next = self.target_net(batch_next_state).detach().gather(1, torch.unsqueeze(argmax, 1))
        q_next = self.target_net(batch_next_state).detach()  # target net is not updated
        q_next = q_next.to(device)

        q_target = batch_reward + GAMMA * q_next.max(1)[0].unsqueeze(1) * (-batch_is_done + 1)

        # q_target = batch_reward + GAMMA * q_next * (-batch_is_done + 1)
        # Loss
        loss = mse_loss(q_eval, q_target)
        # Clear the gradient of parameters when train on different batch
        self.optimizer.zero_grad()
        # Loss backward
        loss.backward()
        # for param in self.eval_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # Update the parameters based on gradient
        self.optimizer.step()
        return loss.item(), avg_q.item()
