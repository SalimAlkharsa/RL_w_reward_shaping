import numpy as np
import argparse
import os

import torch
from agent import DQNAgent, QNetwork, ReplayBuffer
from environment_set_up import EnvironmentSetup
import time
import matplotlib.pyplot as plt
import cv2
import logging  # Add logging module

from helpers import DEVICE, LastVisitedElements, save_model, shape_rewards

# Global variables
DISTANCE_THRESHOLD = 8  # Minimum distance to consider a corner as visited
MAX_STEPS = 5000  # Maximum number of steps per episode
SEED = 42

def visualize_frame(frame, agent_position=None, key_position=None, ladder_position=None ,filename="next_frame.png"):
    logging.info("Visualizing frame...")
    logging.info(f"Frame shape: {frame.shape}")
    
    plt.imshow(frame)
    plt.axis('off')

    # If agent_position is provided, add an 'X' marker
    if agent_position is not None:
        x, y = agent_position
        plt.plot(x, y, marker='x', color='red', markersize=10, label='Agent')
        plt.legend(loc='upper left')  # Add a legend to label the marker

    # If key_position is provided, add a 'O' marker
    if key_position is not None:
        x, y = key_position
        plt.plot(x, y, marker='o', color='yellow', markersize=10, label='Key')
        plt.legend(loc='upper left')  # Add a legend to label the marker
    
    # If ladder_position is provided, add a '+' marker
    if ladder_position is not None:
        x, y = ladder_position
        plt.plot(x, y, marker='+', color='green', markersize=10, label='Ladder')
        plt.legend(loc='upper left')  # Add a legend to label the marker

    # Save the frame with the marker
    plt.savefig(filename)
    print(f"Frame saved as {filename}.")

def identify_agent(frame, game="MontezumaRevenge-v4"):
    # Define the agent's color in RGB format (initial color)
    if game == "MontezumaRevenge-v4":
        agent_color_rgb = np.array([200, 72, 72])  # Red color (target color)
    elif game == "Breakout-v4":
        agent_color_rgb = np.array([213, 130, 74])

    # Convert the frame to HSV color space for better range matching
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Convert the agent's RGB color to HSV
    agent_color_hsv = cv2.cvtColor(np.uint8([[agent_color_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    # Define a range around the agent's HSV color (allowing some tolerance)
    lower_bound = np.array([agent_color_hsv[0] - 10, 50, 50])  # Lower bound of red in HSV with tolerance
    upper_bound = np.array([agent_color_hsv[0] + 10, 255, 255])  # Upper bound of red in HSV with tolerance

    # Create a mask to isolate the red color within the defined range
    mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)

    # Optional: Apply a slight blur to the mask to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)

        # Find the center of the contour
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            return x, y
    return None


def identify_corners(map_resized, graph=False):
    map_resized = cv2.cvtColor(map_resized, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(np.float32(map_resized), blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)  # Dilate to enhance visibility
    threshold = 0.05 * corners.max()
    corner_locations = np.argwhere(corners > threshold)  # Extract corner coordinates

    # Visualize corners on the map
    map_with_corners = cv2.cvtColor(map_resized, cv2.COLOR_GRAY2BGR)
    for y, x in corner_locations:
        cv2.circle(map_with_corners, (x, y), 1, (0, 0, 255), -1)  # Red circle for corners

    # Save or display the map with detected corners
    # Plotting the result using Matplotlib
    # Convert the resized image to RGB for display
    map_resized_rgb = cv2.cvtColor(map_resized, cv2.COLOR_BGR2RGB)

    # Display the image with detected corners
    if graph:
        plt.imshow(map_resized_rgb)
        plt.scatter(corner_locations[:, 1], corner_locations[:, 0], color='red', s=10)  # Plot small red dots
        plt.title("Corners Detected")
        plt.axis('off')  # Turn off the axis for cleaner display
        plt.savefig("corners_detected.png")
        plt.close()

    return corner_locations


def train_dqn(agent, game='MontezumaRevenge-v4', n_episodes=500, target_update_freq=100, render_freq=10, render=False, save_freq=50):
    for episode in range(n_episodes):
        # Reset environment and preprocess state
        state_info = agent.env.reset()
        
        # Extract visual observation and flatten
        
        # Represenring the state as a 32x32 image 
        state = np.array(state_info[0])
        logging.info(f"State shape: {state.shape}")
        
        resized_state = cv2.resize(state, (32, 32))
        # visualize_frame(resized_state, filename="next_frame.png")

        # Identify the corners in the map
        corners = (identify_corners(resized_state, graph=True).tolist())
        # hash the corners to avoid duplicates
        corners = [tuple(corner) for corner in corners]
        corners = set(corners)
        corners_visited = set()  # Track visited corners
        # Track a stack of the last visited corners to penalize (limit size to 3)
        last_visited_corners = LastVisitedElements(max_size=10)

        #### START HERE ####
        total_reward = 0
        done = False

        # Temporarily set render mode to "human" every 10 episodes
        if episode % render_freq == 0 and render and episode > 0:
            print("Rendering the environment...FFF")
            agent.env = EnvironmentSetup(env_name=game, render_mode="human").env
            agent.env.reset()
        else:
            agent.env = EnvironmentSetup(env_name=game, render_mode=None).env
            agent.env.reset()
        agent.env.action_space.seed(SEED)
        agent.env.seed(SEED)
        agent.env.observation_space.seed(SEED)


        # Training loop 
        actions = 0   
        while not done:
            actions += 1
            if actions > MAX_STEPS:
                logging.info("Max steps reached")
                break
            
            
            # Select action
            # action = agent.act(flattened_state)
            action = agent.act(resized_state)
            
            # Take action in the environment
            next_state, reward, done, truncated, info = agent.env.step(action)  # Correct unpacking

            # Reformat the next state's map
            next_state = np.array(next_state)
            next_state = cv2.resize(next_state, (32, 32))

            # Locate the agent in the frame (remember we are working with the resized frame)
            agent_position = identify_agent(next_state, game=game)
            
            # Apply the rewards shaping function here: (commented out for baseline)
            # reward += shape_rewards(agent_position, corners, corners_visited, last_visited_corners, DISTANCE_THRESHOLD)
        
            # Flatten prior to passing to the QNetwork
            flattened_next_state = np.array(next_state).flatten()

            # agent.replay_buffer.store((flattened_state, action, reward, flattened_next_state, done))
            agent.replay_buffer.store((resized_state, action, reward, next_state, done))
            # Learn from replay buffer
            agent.replay()
            
            # Update state
            flattened_state = flattened_next_state
            resized_state = next_state
            total_reward += reward

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_model()
        
        # Save the model every save_freq episodes
        if episode % save_freq == 0:
            save_model(agent, episode, save_dir=f"models_{game}")
        
        logging.info(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}")
        
        # Optionally close the environment after training
        if episode == n_episodes - 1:
            agent.env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Montezuma's Revenge or CartPole-v1")
    parser.add_argument('--env', type=str, default="MontezumaRevenge-v4", choices=["MontezumaRevenge-v4", "CartPole-v1", "Breakout-v4"], help="Environment to train on")
    parser.add_argument('--n_episodes', type=int, default=5000, help="Number of episodes to train the agent")
    parser.add_argument('--target_update_freq', type=int, default=100, help="Frequency of updating target network")
    parser.add_argument('--render', action="store_true", help="Render the environment")
    parser.add_argument('--render_freq', type=int, default=50, help="Frequency of rendering the environment")
    parser.add_argument('--save_freq', type=int, default=2, help="Frequency of saving the model")
    args = parser.parse_args()


    # Set logging
    # Configure logging
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    # Initialize environment
    env_setup = EnvironmentSetup(env_name=args.env, render_mode= None)#"human" if args.render else None)
    obs_shape = env_setup.env.observation_space.shape
    n_actions = env_setup.env.action_space.n

    # Flattened input dimension
    input_dim = np.prod(obs_shape)
    input_dim = (3, 32, 32) # ---> due to the resizing of the frame

    # Initialize models
    # Device setup for Mac (MPS), CUDA, or CPU
    device = DEVICE
    logging.info(f"Using device: {device}")
    q_model = QNetwork(input_dim, n_actions).to(device)
    target_model = QNetwork(input_dim, n_actions).to(device)
    target_model.load_state_dict(q_model.state_dict())  # Synchronize weights

    # Replay buffer
    replay_buffer = ReplayBuffer()

    # Initialize agent
    agent = DQNAgent(env_setup.env, q_model, target_model, replay_buffer)

    # Train the agent
    train_dqn(agent, game=args.env, n_episodes=args.n_episodes, target_update_freq=args.target_update_freq, render_freq=args.render_freq,
              render=args.render, save_freq=args.save_freq)
