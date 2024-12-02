from collections import deque
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from helpers import DEVICE

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, latent_size=1024):
        super(QNetwork, self).__init__()

        # First transformation: Linear layer followed by activation (f1)
        self.fc1 = nn.Linear(state_size, 5012)  # First linear transformation (sW1 + b1)
        
        # Second transformation: Linear layer followed by activation (f2)
        self.fc2 = nn.Linear(5012, latent_size)  # Second linear transformation (f1 * W2 + b2)
        
        # Third transformation: Linear layer followed by activation (f3)
        self.fc3 = nn.Linear(latent_size, action_size)  # Final output layer (f2 * W3 + b3)

    def forward(self, state):
        # Apply the transformations with activations between each layer as per f1, f2, f3
        x = F.relu(self.fc1(state))  # Apply f1 activation after fc1
        x = F.relu(self.fc2(x))  # Apply f2 activation after fc2
        q_values = self.fc3(x)  # Apply fc3 to get the final Q-values
        
        return q_values


class ReplayBuffer:
    def __init__(self, capacity=10**4):
        self.memory = deque(maxlen=capacity)
    
    def store(self, experience):
        state, action, reward, next_state, done = experience
        self.memory.append((
        np.array(state, dtype=np.float32),
        int(action),
        float(reward),
        np.array(next_state, dtype=np.float32),
        bool(done)
        ))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def size(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, env, q_model, target_model, replay_buffer, batch_size=32, gamma=0.995, epsilon=1.0, 
                 epsilon_min=0.1, epsilon_decay=0.99, learning_rate=1e-4):
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
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.q_model(state)
        return torch.argmax(q_values).item()
    
    def replay(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert sampled data to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)  # Ensure actions are integers
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)  # Ensure rewards are floats
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)  # Ensure dones are booleans

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
