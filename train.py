import gym
import numpy as np
import argparse
import os
import torch
from agent import DQNAgent, QNetwork, ReplayBuffer
from environment_set_up import EnvironmentSetup
import time
import matplotlib.pyplot as plt
import cv2
import logging  # Logging module
from helpers import DEVICE, LastVisitedElements, save_model, shape_rewards

# Global variables
SMOOTHING_FACTOR = 0.1 # Smoothing factor for the extrinsic reward
DISTANCE_THRESHOLD = 8  # Minimum distance to consider a corner as visited
MAX_STEPS = 5000  # Maximum number of steps per episode
SEED = 42


def visualize_frame(frame, agent_position=None, key_position=None, ladder_position=None, filename="next_frame.png"):
    logging.info("Visualizing frame...")
    plt.imshow(frame)
    plt.axis('off')

    # Add markers if positions are provided
    if agent_position:
        plt.plot(*agent_position, marker='x', color='red', markersize=10, label='Agent')
    if key_position:
        plt.plot(*key_position, marker='o', color='yellow', markersize=10, label='Key')
    if ladder_position:
        plt.plot(*ladder_position, marker='+', color='green', markersize=10, label='Ladder')

    # Save and log the frame
    plt.legend(loc='upper left')
    plt.savefig(filename)
    plt.close()
    logging.info(f"Frame saved as {filename}.")


def identify_agent(frame, game="MontezumaRevenge-v4"):
    """Identify agent in the frame based on the game and specific color."""
    game_colors = {
        "MontezumaRevenge-v4": [200, 72, 72],
        "Breakout-v4": [213, 130, 74],
        "BankHeist-v4": [76, 46, 15],
    }
    agent_color_rgb = np.array(game_colors.get(game, [0, 0, 0]))  # Default to black if game not listed
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    agent_color_hsv = cv2.cvtColor(np.uint8([[agent_color_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    lower_bound = np.array([agent_color_hsv[0] - 10, 50, 50])
    upper_bound = np.array([agent_color_hsv[0] + 10, 255, 255])
    mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        if M["m00"]:
            return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return None


def identify_corners(map_resized, graph=False):
    """Identify corners in the resized map."""
    gray_map = cv2.cvtColor(map_resized, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(np.float32(gray_map), blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)
    threshold = 0.05 * corners.max()
    corner_locations = np.argwhere(corners > threshold)

    if graph:
        plt.imshow(cv2.cvtColor(map_resized, cv2.COLOR_BGR2RGB))
        plt.scatter(corner_locations[:, 1], corner_locations[:, 0], color='red', s=10)
        plt.title("Corners Detected")
        plt.axis('off')
        plt.savefig("corners_detected.png")
        plt.close()

    return corner_locations


def train_dqn(agent, game='MontezumaRevenge-v4', n_episodes=500, target_update_freq=100, render_freq=10, render=False, save_freq=50, extrinsic=False):
    for episode in range(n_episodes):
        # Reset environment and preprocess state
        state_info = agent.env.reset()
        state = np.array(state_info[0])
        logging.info(f"Initial state shape: {state.shape}")

        # Identify the corners in the map (if needed)
        if extrinsic:
            corners = identify_corners(state)
            extrinsic_reward = 0
        else:
            corners = []
        corners_visited = []
        last_visited_corners = LastVisitedElements(max_size=10)

        total_reward = 0
        last_smoothed_reward = 0
        done = False

        # Temporarily set render mode every `render_freq` episodes
        if episode % render_freq == 0 and render:# and episode > 0:
            agent.env = EnvironmentSetup(env_name=game, render_mode="human").env
            agent.env.reset()
        else:
            agent.env = EnvironmentSetup(env_name=game, render_mode=None).env
            agent.env.reset()

        if game not in ["CartPole-v1"]:
            agent.env.action_space.seed(SEED)
            agent.env.seed(SEED)
            agent.env.observation_space.seed(SEED)
            # Resize the state for games into a 64x64 image
            state = cv2.resize(state, (32, 32), interpolation=cv2.INTER_LINEAR)


        # Flatten the state
        flattened_state = state.flatten()

        # Training loop
        steps = 0
        while not done:
            steps += 1
            if steps > MAX_STEPS and game not in ["CartPole-v1"]:
                logging.info("Max steps reached")
                break
            
            # Select action
            action = agent.act(flattened_state)

            # Take action in the environment
            next_state, reward, done, truncated, info = agent.env.step(action)

            # Resize the state for games into a 64x64 image
            if game not in ["CartPole-v1"]:
                next_state = cv2.resize(next_state, (32, 32), interpolation=cv2.INTER_LINEAR)

            # Flatten the next state
            flattened_next_state = np.array(next_state).flatten()

            # Apply reward function
            if extrinsic:
                # get agent position    
                agent_position = identify_agent(next_state, game)
                # Calculate the reward
                extrinsic_reward, corners_visited, last_visited_corners = shape_rewards(agent_position, corners, corners_visited, last_visited_corners, DISTANCE_THRESHOLD)
                extrinsic_reward = SMOOTHING_FACTOR * extrinsic_reward + (1 - SMOOTHING_FACTOR) * last_smoothed_reward
                last_smoothed_reward = extrinsic_reward







            # Store the experience in replay buffer
            agent.replay_buffer.store((flattened_state, action, reward, flattened_next_state, done))

            # Perform a replay step
            agent.replay()

            # Update state
            flattened_state = flattened_next_state

            if truncated or done:
                logging.info("Truncated episode")
                break

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_model()

        # Save the model every `save_freq` episodes
        if episode % save_freq == 0:
            save_model(agent, episode, save_dir=f"models_{game}")
        
        if extrinsic:
            logging.info(f"Intrinsic reward: {reward}")
            logging.info(f"Extrinsic reward: {extrinsic_reward}")
            total_reward += extrinsic_reward

        logging.info(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}")

        # Optionally close the environment after training
        if episode == n_episodes - 1:
            agent.env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Montezuma's Revenge or CartPole-v1")
    parser.add_argument('--env', type=str, default="MontezumaRevenge-v4", choices=["MontezumaRevenge-v4", "CartPole-v1", "Breakout-v4", "BankHeist-v4"], help="Environment to train on")
    parser.add_argument('--n_episodes', type=int, default=5000, help="Number of episodes to train the agent")
    parser.add_argument('--target_update_freq', type=int, default=100, help="Frequency of updating target network")
    parser.add_argument('--render', action="store_true", help="Render the environment")
    parser.add_argument('--render_freq', type=int, default=50, help="Frequency of rendering the environment")
    parser.add_argument('--save_freq', type=int, default=2, help="Frequency of saving the model")
    parser.add_argument('--extrinsic', action="store_true", help="Use extrinsic reward shaping")
    args = parser.parse_args()

    # Set logging
    logging.basicConfig(filename='CartPole.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    # Initialize environment
    env_setup = EnvironmentSetup(env_name=args.env, render_mode=None)
    obs_shape = env_setup.env.observation_space.shape
    n_actions = env_setup.env.action_space.n

    # Adjust input dimensions to match the environment's observation space
    input_dim = np.prod(obs_shape)
    if args.env not in ["CartPole-v1"]:
        input_dim =  32 * 32 * 3

    # Initialize models
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
              render=args.render, save_freq=args.save_freq, extrinsic=args.extrinsic)
