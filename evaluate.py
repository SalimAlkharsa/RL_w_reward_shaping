import numpy as np
import argparse
import torch
import cv2
import matplotlib.pyplot as plt
import logging

from agent import DQNAgent, QNetwork, ReplayBuffer
from environment_set_up import EnvironmentSetup
from helpers import DEVICE, LastVisitedElements, shape_rewards
from train import DISTANCE_THRESHOLD, SEED, identify_agent, identify_corners

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

def evaluate_dqn(agent, game='MontezumaRevenge-v4', episode=0):
    
    state_info = agent.env.reset()
    state = np.array(state_info[0])
    resized_state = cv2.resize(state, (32, 32))
    total_reward = 0
    done = False

    # Identify the corners in the map
    corners = (identify_corners(resized_state, graph=True).tolist())
    # hash the corners to avoid duplicates
    corners = [tuple(corner) for corner in corners]
    corners = set(corners)
    corners_visited = set()  # Track visited corners
    # Track a stack of the last visited corners to penalize (limit size to 3)
    last_visited_corners = LastVisitedElements(max_size=10)

    while not done:
        action = agent.act(resized_state)
        next_state, reward, done, truncated, info = agent.env.step(action)
        next_state = np.array(next_state)
        next_state = cv2.resize(next_state, (32, 32))
        agent_position = identify_agent(next_state, game=game)
        reward += shape_rewards(agent_position, corners, corners_visited, last_visited_corners, DISTANCE_THRESHOLD)
        resized_state = next_state
        total_reward += reward

    logging.info(f"Episode {episode+1}/{episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN on Montezuma's Revenge or CartPole-v1")
    parser.add_argument('--env', type=str, default="MontezumaRevenge-v4", choices=["MontezumaRevenge-v4", "CartPole-v1", "Breakout-v4"], help="Environment to evaluate on")
    parser.add_argument('--episode', type=int, default=0, help="Number of episodes to evaluate the agent")
    parser.add_argument('--render', action='store_true', help="Render the environment")
    args = parser.parse_args()

    # Set the logging level
    logging.basicConfig(filename='evaluation.log', level=logging.INFO, format='%(asctime)s - %(message)s')


    # Set the model path based on the environment
    model_path = f"models_{args.env}/model_episode_{args.episode}.pth"

    env_setup = EnvironmentSetup(env_name=args.env, render_mode="human" if args.render else None)
    obs_shape = env_setup.env.observation_space.shape
    n_actions = env_setup.env.action_space.n
    input_dim = (3, 32, 32)

    device = DEVICE
    q_model = QNetwork(input_dim, n_actions).to(device)
    target_model = QNetwork(input_dim, n_actions).to(device)
    # Load the model
    checkpoint = torch.load(model_path, map_location=DEVICE)
    q_model.load_state_dict(checkpoint['state_dict'])
    target_model.load_state_dict(checkpoint['target_state_dict'])

    replay_buffer = checkpoint['replay_buffer']
    agent = DQNAgent(env_setup.env, q_model, target_model, replay_buffer)
    agent.epsilon = checkpoint['epsilon']

    agent.env.action_space.seed(SEED)
    agent.env.seed(SEED)
    agent.env.observation_space.seed(SEED)

    evaluate_dqn(agent, game=args.env, episode=args.episode)