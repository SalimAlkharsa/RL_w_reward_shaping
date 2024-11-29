import numpy as np
from agent import DQNAgent, QNetwork, ReplayBuffer
from environment_set_up import EnvironmentSetup


import time

def train_dqn(agent, n_episodes=500, target_update_freq=1000, render_freq=500):
    for episode in range(n_episodes):
        # Reset environment and preprocess state
        state_info = agent.env.reset()
        
        # Extract visual observation and flatten
        state = np.array(state_info[0]).flatten()  # Use the first element (visual observation)

        total_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action in the environment
            next_state, reward, done, truncated, info = agent.env.step(action)  # Correct unpacking
            
            # Extract and flatten the next state
            next_state = np.array(next_state[0]).flatten()  # Extract and flatten visual frame
            if state.shape != next_state.shape:
                continue
            # Check the shape after flattening
            print(f"Next state shape after flattening: {next_state.shape}")
            print(f"Next state after flattening: {state.shape}")
            
            # Store experience in replay buffer
            agent.replay_buffer.store((state, action, reward, next_state, done))

            # Learn from replay buffer
            agent.replay()
            
            # Update state
            state = next_state
            total_reward += reward
            
            # Render environment periodically
            if episode % render_freq == 0:
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
    # Initialize environment
    env_setup = EnvironmentSetup()
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
    train_dqn(agent, n_episodes=5000)