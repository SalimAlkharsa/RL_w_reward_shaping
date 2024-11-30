import numpy as np
import argparse
from agent import DQNAgent, QNetwork, ReplayBuffer
from environment_set_up import EnvironmentSetup
import time
import matplotlib.pyplot as plt
import cv2

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
    """
    Identify the red blob (agent) in the frame and return its location.
    Args:
        frame (np.ndarray): The game frame (RGB format).
    Returns:
        tuple: (x, y) coordinates of the agent's center or None if not found.
    """
    # Define the agent's color in RGB format
    agent_color = np.array([200,72,72]) # Red color

    # Find the agent's location in the frame
    mask = cv2.inRange(frame, agent_color, agent_color)
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

def identify_key(frame):
    base_key_color = np.array([228, 200, 97])  # Yellow color

    # Let give some tolerance for the key color
    tolerance = 50
    key_color_lb = np.array([base_key_color[0] - tolerance, base_key_color[1] - tolerance, base_key_color[2] - tolerance])
    key_color_ub = np.array([base_key_color[0] + tolerance, base_key_color[1] + tolerance, base_key_color[2] + tolerance])


    mask = cv2.inRange(frame, key_color_lb, key_color_ub)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)

        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            return x, y
        
    return None

def identify_closest_ladder(frame, agent_position):
    # Ladder rung color (adjust based on the rung pattern)
    base_ladder_colors = [
        np.array([0, 0, 0]),        # Color 1
        np.array([31, 75, 61]),     # Rung 1
        np.array([66, 158, 130])  # Rung 2
        # np.array([60, 145, 119]),   # Rung 3
        # np.array([9, 22, 18]),      # Rung 4
        # np.array([0, 0, 0])         # Rung 5
    ]
    
    # Tolerance for the color detection
    tolerance = 70  # Adjust tolerance to capture color variations
    ladder_mask = np.zeros_like(frame[:, :, 0])  # To store the mask

    # Create a mask by checking all the rung colors
    for color in base_ladder_colors:
        color_lb = np.array([color[0] - tolerance, color[1] - tolerance, color[2] - tolerance])
        color_ub = np.array([color[0] + tolerance, color[1] + tolerance, color[2] + tolerance])

        # Create a mask for the current color range
        mask = cv2.inRange(frame, color_lb, color_ub)
        ladder_mask = cv2.bitwise_or(ladder_mask, mask)  # Combine all masks

    # Find contours in the mask
    contours, _ = cv2.findContours(ladder_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # List to hold possible ladder rungs (each rung is a set of vertically stacked blocks)
    possible_ladders = []

    # Group the blocks based on vertical stacking
    for contour in contours:
        M = cv2.moments(contour)

        if M["m00"] != 0:
            # Calculate the center of mass of the contour
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Draw a circle at the center of mass of each rung (block)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Green circle for rungs

            # Check if this is part of an existing ladder (check if it's vertically aligned with other rungs)
            matched = False
            for ladder in possible_ladders:
                for (lx, ly) in ladder:
                    if abs(lx - cx) < 10:  # If they're close in the x-direction
                        ladder.append((cx, cy))  # Add to the existing ladder
                        matched = True
                        break
                if matched:
                    break

            # If not matched with any ladder, start a new ladder
            if not matched:
                possible_ladders.append([(cx, cy)])

    # Now draw the centers of the detected ladders
    for ladder in possible_ladders:
        # For each ladder, take the average position of its rungs (stacked blocks)
        ladder_x = np.mean([cx for cx, cy in ladder])
        ladder_y = np.mean([cy for cx, cy in ladder])

        # Draw a larger circle to mark the center of the ladder
        cv2.circle(frame, (int(ladder_x), int(ladder_y)), 10, (0, 0, 255), -1)  # Red circle for ladder center

    # Show the final image with the possible ladders marked
    # cv2.imshow("Detected Ladders", frame)
    # cv2.waitKey(0)  # Wait for a key press to close the image
    # cv2.destroyAllWindows()

    # Find the closest ladder
    closest_ladder_center = None
    closest_distance = float('inf')

    for ladder in possible_ladders:
        # For each ladder, take the average position of its rungs (stacked blocks)
        ladder_x = np.mean([cx for cx, cy in ladder])
        ladder_y = np.mean([cy for cx, cy in ladder])

        # Calculate the distance to the agent's position
        distance = np.sqrt((ladder_x - agent_position[0])**2 + (ladder_y - agent_position[1])**2)

        if distance < closest_distance:
            closest_ladder_center = (ladder_x, ladder_y)
            closest_distance = distance

    return closest_ladder_center, ladder_mask

def train_dqn(agent, n_episodes=500, target_update_freq=100, render_freq=5000, render=False, timeout=90):
    visited_states = {}
    for episode in range(n_episodes):
        # Reset environment and preprocess state
        state_info = agent.env.reset()
        
        # Extract visual observation and flatten
        state = np.array(state_info[0]).flatten()  # Use the first element (visual observation)

        total_reward = 0
        done = False
        start_time = time.time()        
        while not done:
            # Check for timeout
            if time.time() - start_time > timeout:
                print(f"Episode {episode+1}: Timeout reached, skipping the rest of the episode.")
                break
            
            # Select action
            action = agent.act(state)
            
            # Take action in the environment
            next_state, reward, done, truncated, info = agent.env.step(action)  # Correct unpacking

            # Extract and flatten the next state
            # next_state = np.array(next_state).flatten()  # Extract and flatten visual frame

            # Visualize the frame with agent and key positions
            agent_position = identify_agent(next_state)
            key_position = identify_key(next_state)
            ladder_position, _ = identify_closest_ladder(next_state, agent_position)
            #visualize_frame(next_state, agent_position=agent_position, key_position=key_position, ladder_position=ladder_position,
            #                filename="next_frame.png")

            flattened_next_state = tuple(np.array(next_state).flatten())  # Convert to hashable type
            if flattened_next_state not in visited_states:
                visited_states[flattened_next_state] = 1
                reward += 0.0  # Reward for visiting a new state
            else:
                visited_states[flattened_next_state] += 1
                reward -= 0.0  # Small penalty for revisiting a state
                
            # Reward the agent if it gets closer to the key
            if agent_position and key_position:
                agent_x, agent_y = agent_position
                key_x, key_y = key_position
                distance = np.sqrt((agent_x - key_x) ** 2 + (agent_y - key_y) ** 2)
                # Reward the agent for getting closer to the ladder too
                if ladder_position:
                    ladder_x, ladder_y = ladder_position
                    ladder_distance = np.sqrt((agent_x - ladder_x) ** 2 + (agent_y - ladder_y) ** 2)
                    reward += (1 - ladder_distance / 100)*2
                reward += 1 - distance / 100

            # flatten the next state?
            next_state = np.array(next_state).flatten()
            # Store experience in replay buffer
            agent.replay_buffer.store((state, action, reward, next_state, done))

            if agent_position and key_position:
                print(f"Agent position: {agent_position}, Key position: {key_position}, Ladder position: {ladder_position}")
                print(f"Reward: {reward}")
                print(f"Info: {info}")
                print('--------------------------------------------------------------------')            


            # Learn from replay buffer
            agent.replay()
            
            # Update state
            state = next_state
            total_reward += reward
            
            # Render environment periodically
            if render and episode % render_freq == 0:
                agent.env.render()
                time.sleep(0.1)  # Slow down the rendering for better visibility

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
    parser.add_argument('--timeout', type=int, default=240, help="Timeout for each episode in seconds")
    args = parser.parse_args()

    # Initialize environment
    env_setup = EnvironmentSetup(env_name=args.env, render_mode="human" if args.render else None)
    obs_shape = env_setup.env.observation_space.shape
    n_actions = env_setup.env.action_space.n

    # Flattened input dimension
    input_dim = np.prod(obs_shape)

    # Initialize models
    q_model = QNetwork(input_dim, n_actions)
    target_model = QNetwork(input_dim, n_actions)
    target_model.load_state_dict(q_model.state_dict())  # Synchronize weights

    # Replay buffer
    replay_buffer = ReplayBuffer()

    # Initialize agent
    agent = DQNAgent(env_setup.env, q_model, target_model, replay_buffer)

    # Train the agent
    train_dqn(agent, n_episodes=args.n_episodes, target_update_freq=args.target_update_freq, render=args.render, timeout=args.timeout)
