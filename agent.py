from collections import deque
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import random

class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, n_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)


class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.memory = deque(maxlen=capacity)
    
    def store(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def size(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, env, q_model, target_model, replay_buffer, batch_size=32, gamma=0.99, epsilon=1, epsilon_min=0.1, epsilon_decay=0.995, learning_rate=1e-4):
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
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_model(state)
        return torch.argmax(q_values).item()
    
    def replay(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        # Compute Q(s, a)
        q_values = self.q_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
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
