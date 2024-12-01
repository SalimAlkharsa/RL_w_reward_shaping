import numpy as np
import argparse
from agent import DQNAgent, QNetwork, ReplayBuffer
from environment_set_up import EnvironmentSetup
import time
import matplotlib.pyplot as plt
import cv2

DISTANCE_THRESHOLD = 4  # Minimum distance to consider a corner as visited

def visualize_frame(frame, agent_position=None, key_position=None, ladder_position=None ,filename="next_frame.png"):
    print("Visualizing frame...")
    print(frame.shape)
    
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

def identify_agent(frame):
    # Define the agent's color in RGB format (initial color)
    agent_color_rgb = np.array([200, 72, 72])  # Red color (target color)

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


def train_dqn(agent, game='MontezumaRevenge-v4', n_episodes=500, target_update_freq=100, render_freq=10, render=False):
    for episode in range(n_episodes):
        # Reset environment and preprocess state
        state_info = agent.env.reset()
        
        # Extract visual observation and flatten
        
        # Represenring the state as a 32x32 image 
        state = np.array(state_info[0])
        print(f"State shape: {state.shape}")
        
        resized_state = cv2.resize(state, (32, 32))
        # visualize_frame(resized_state, filename="next_frame.png")
        flattened_state = np.array(resized_state).flatten() # --> need that to pass it to the QNetwork
        print(f"Flattened state shape: {flattened_state.shape}")

        # Identify the corners in the map
        corners = (identify_corners(resized_state, graph=True).tolist())
        # hash the corners to avoid duplicates
        corners = [tuple(corner) for corner in corners]
        corners = set(corners)
        corners_visited = set()  # Track visited corners

        ######## DEBUGGING ########
        print(f"Agent starting position: {identify_agent(resized_state)}")
        print

        ######## DEBUGGING ########

        #### START HERE ####
        total_reward = 0
        done = False

        # Temporarily set render mode to "human" every 10 episodes
        if episode % render_freq == 0 and render and episode > 0:
            agent.env = EnvironmentSetup(env_name=game, render_mode="human").env
            agent.env.reset()
        else:
            agent.env = EnvironmentSetup(env_name=game, render_mode=None).env
            agent.env.reset()


        # Training loop    
        while not done:
            # Render the environment every 10 episodes
            # if render and episode % render_freq == 0:
            #     agent.env.render(mode="human")
            
            # Select action
            action = agent.act(flattened_state)
            
            # Take action in the environment
            next_state, reward, done, truncated, info = agent.env.step(action)  # Correct unpacking

            # Reformat the next state's map
            next_state = np.array(next_state)
            next_state = cv2.resize(next_state, (32, 32))

            # visualize_frame(next_state, filename="next_frame.png")
            

            # Locate the agent in the frame (remember we are working with the resized frame)
            agent_position = identify_agent(next_state)
            # Update the visited corners based on the agent's position
            # Iterate over each corner and check if the agent is close enough
            if agent_position is not None:
                for corner in corners:
                    # Calculate the Euclidean distance between the agent and the corner
                    distance = np.sqrt((corner[0] - agent_position[0])**2 + (corner[1] - agent_position[1])**2)
                    
                    # If the distance is within the threshold, mark the corner as visited
                    if distance <= DISTANCE_THRESHOLD and corner not in corners_visited:
                        corners_visited.add(corner)
                        # Reward the agent for visiting a new corner
                        reward += 1

                    # Let us also reward the agent for getting closer to UNVISITED corners
                    elif corner not in corners_visited:
                        reward += 0.1 / distance


            # Flatten prior to passing to the QNetwork
            flattened_next_state = np.array(next_state).flatten()

            agent.replay_buffer.store((flattened_state, action, reward, flattened_next_state, done))
            # Learn from replay buffer
            agent.replay()
            
            # Update state
            flattened_state = flattened_next_state
            total_reward += reward

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_model()
        
        print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}")
        
        # Optionally close the environment after training
        if episode == n_episodes - 1:
            agent.env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Montezuma's Revenge or CartPole-v1")
    parser.add_argument('--env', type=str, default="MontezumaRevenge-v4", choices=["MontezumaRevenge-v4", "CartPole-v1"], help="Environment to train on")
    parser.add_argument('--n_episodes', type=int, default=5000, help="Number of episodes to train the agent")
    parser.add_argument('--target_update_freq', type=int, default=100, help="Frequency of updating target network")
    parser.add_argument('--render', action="store_true", help="Render the environment")
    parser.add_argument('--render_freq', type=int, default=10, help="Frequency of rendering the environment")
    args = parser.parse_args()

    # Initialize environment
    env_setup = EnvironmentSetup(env_name=args.env, render_mode= None)#"human" if args.render else None)
    obs_shape = env_setup.env.observation_space.shape
    n_actions = env_setup.env.action_space.n

    # Flattened input dimension
    input_dim = np.prod(obs_shape)
    input_dim = 32*32*3 # ---> due to the resizing of the frame

    # Initialize models
    q_model = QNetwork(input_dim, n_actions)
    target_model = QNetwork(input_dim, n_actions)
    target_model.load_state_dict(q_model.state_dict())  # Synchronize weights

    # Replay buffer
    replay_buffer = ReplayBuffer()

    # Initialize agent
    agent = DQNAgent(env_setup.env, q_model, target_model, replay_buffer)

    # Train the agent
    train_dqn(agent, game=args.env, n_episodes=args.n_episodes, target_update_freq=args.target_update_freq, render_freq=args.render_freq,
              render=args.render)
