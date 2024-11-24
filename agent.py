import random
import numpy as np
import torch.nn as nn
from collections import deque
import torch.optim as optim

import torch

from model_bacbones.simple_cnn import SimpleCNN as DQN

class DQNAgent:
    def __init__(self, env, model, batch_size=32, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=1e-4):
        self.env = env
        self.model = model
        self.target_model = DQN(env.observation_space.shape, env.action_space.n)  # Target model for stability
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=2000)
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()  # Exploration: random action
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state)
        return torch.argmax(q_values).item()  # Exploitation: choose action with max Q value
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        # Compute Q values for current states
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values for next states
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        # Loss function (Mean Squared Error)
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Reduce epsilon (decay)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
