import matplotlib.pyplot as plt
import subprocess

def parse_log_file(log_file):
    episodes = []
    rewards = []

    # Use grep to extract lines containing "Total Reward"
    result = subprocess.run(['grep', 'Total Reward', log_file], capture_output=True, text=True)
    lines = result.stdout.splitlines()

    for line in lines:
        parts = line.strip().split(' - ')
        episode_info = parts[1].split(',')[0]
        reward_info = parts[1].split(',')[1]
        episode = int(episode_info.split()[1].split('/')[0])
        reward = float(reward_info.split()[-1])
        episodes.append(episode)
        rewards.append(reward)

    return episodes, rewards

def plot_rewards(episodes, rewards):
    plt.plot(episodes, rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    log_file = 'training.log'
    episodes, rewards = parse_log_file(log_file)
    plot_rewards(episodes, rewards)