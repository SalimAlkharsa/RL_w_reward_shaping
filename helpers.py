from collections import deque
import logging
import os
import numpy as np
import torch


# Define the device to be used, does not really work right now
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Use MPS if available
    DEVICE = torch.device("cpu") # Default to CPU even w MPS due to time constraints
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # Use CUDA if available
else:
    DEVICE = torch.device("cpu")  # Default to CPU if no GPU is available

# Create a class to store the last visited elements
class LastVisitedElements:
    def __init__(self, max_size=3):
        self.max_size = max_size
        self.elements = deque(maxlen=max_size)

    def add_element(self, element):
        self.elements.append(element)

    def get_elements(self):
        return set(self.elements)

# Helper function to shape the rewards
def shape_rewards(agent_position, corners, corners_visited, last_visited_corners, DISTANCE_THRESHOLD=1):
    reward = 0
    # Iterate over each corner and check if the agent is close enough
    if agent_position is not None and len(corners) > 0:
        for corner in corners:
            # Calculate the Euclidean distance between the agent and the corner
            distance = np.sqrt((corner[0] - agent_position[0])**2 + (corner[1] - agent_position[1])**2)
            
            # If the distance is within the threshold, mark the corner as visited
            if distance <= DISTANCE_THRESHOLD and corner not in corners_visited:
                corners_visited.add(corner)
                # Reward the agent for visiting a new corner
                reward += 1
                # Update the last visited corners stack
                last_visited_corners.add_element(corner)
                

            # Let us also reward the agent for getting closer to UNVISITED corners
            elif corner not in corners_visited:
                reward += 0.1 / distance

            # Penalize the agent for going to a cornner it just visited
            elif distance <= DISTANCE_THRESHOLD and corner in last_visited_corners.get_elements():
                reward -= 0.1
    return reward, corners_visited, last_visited_corners

# Helper function to save the model
def save_model(agent, episode, save_dir="default_save"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, f"model_episode_{episode}.pth")
    torch.save({
        'state_dict': agent.q_model.state_dict(),
        'target_state_dict': agent.target_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode': episode,
        'replay_buffer': agent.replay_buffer,
        'epsilon': agent.epsilon,  # Save exploration rate if using epsilon-greedy
     }, f"{save_dir}/model_episode_{episode}.pth")