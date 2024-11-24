import numpy as np

from agent import DQNAgent
from environment_set_up import EnvironmentSetup
from model_bacbones.simple_cnn import SimpleCNN


def train_dqn(agent, n_episodes=1000):
    for episode in range(n_episodes):
        state = agent.env.reset()
        
        # Check if the state is a tuple (which it often is in Gym environments)
        if isinstance(state, tuple):
            state = state[0]  # Extract the actual state from the tuple
        
        state = np.array(state)
        total_reward = 0
        
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = agent.env.step(action)

            # Make sure to cast the types as bool when applicable
            terminated = bool(terminated)
            truncated = bool(truncated)
            
            # Handle the new 'done' flag logic in Gym v0.26+ 
            done = terminated or truncated
            
            # Again, check if next_state is a tuple
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # Extract the actual state from the tuple
            
            next_state = np.array(next_state)
            
            # Store experience in replay memory
            agent.remember(state, action, reward, next_state, done)
            
            # Replay and learn
            agent.replay()
            state = next_state
            total_reward += reward
        
        agent.update_target_model()
        print(f"Episode {episode+1}/{n_episodes} - Total Reward: {total_reward}")

    agent.env.close()



# Initialize environment, model, and agent
env_setup = EnvironmentSetup()
print(env_setup.env.observation_space.shape)
model = SimpleCNN(env_setup.env.observation_space.shape, env_setup.env.action_space.n)
agent = DQNAgent(env_setup.env, model)

# Train the agent
train_dqn(agent)
