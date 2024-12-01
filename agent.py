from collections import deque
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import random

from helpers import DEVICE

class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(QNetwork, self).__init__()
        # print(input_dim)
        c, h, w = input_dim
        self.conv1 = nn.Conv2d(in_channels=c, out_channels= 32, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, n_actions)

        # Compute flattened size
        convw = ((w - 4) // 1 + 1)  # After Conv1
        convw = ((convw - 4) // 2 + 1)  # After Conv2
        convw = ((convw - 3) // 1 + 1)  # After Conv3

        convh = ((h - 4) // 1 + 1)  # After Conv1
        convh = ((convh - 4) // 2 + 1)  # After Conv2
        convh = ((convh - 3) // 1 + 1)  # After Conv3

        conv_output_size = convw * convh * 64  # 64 is the output channels of Conv3

        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.to(self.conv1.weight.device).to(self.conv1.weight.dtype) 
        # rearrange x to be BCHW
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity=10**6):
        self.memory = deque(maxlen=capacity)
    
    def store(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def size(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, env, q_model, target_model, replay_buffer, batch_size=32, gamma=0.99, epsilon=1, epsilon_min=0.1, epsilon_decay=0.99, learning_rate=1e-3):
        self.env = env
        self.q_model = q_model
        self.target_model = target_model
        self.replay_buffer = replay_buffer
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=learning_rate)
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()  # Random action
        state = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_model(state)
        return torch.argmax(q_values).item()
    
    def replay(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to np arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)

        # Compute Q(s, a)
        q_values = self.q_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        # put gamma on the device of the next_states
        self.gamma = torch.tensor(self.gamma, dtype=torch.float32, device=next_states.device).clone().detach()
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
            # put max_next_q_values on the device of rewards
            max_next_q_values = torch.tensor(max_next_q_values, dtype=torch.float32, device=rewards.device).clone().detach()
            # do the operation
            target_q_values = rewards + self.gamma * max_next_q_values * (~dones)
        
        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # Optimize the Q-Network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.q_model.state_dict())
