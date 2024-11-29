import numpy as np
import argparse
from agent import DQNAgent, QNetwork, ReplayBuffer
from environment_set_up import EnvironmentSetup
import time

def train_dqn(agent, n_episodes=500, target_update_freq=100, render_freq=5000, render=False):
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
            next_state = np.array(next_state).flatten()  # Extract and flatten visual frame
            
            # Store experience in replay buffer
            agent.replay_buffer.store((state, action, reward, next_state, done))

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
    train_dqn(agent, n_episodes=args.n_episodes, target_update_freq=args.target_update_freq, render=args.render)
