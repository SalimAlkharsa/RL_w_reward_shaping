# Reinforcement Learning with Reward Shaping

This repository contains code for training a DQN agent with reward shaping on various environments including Montezuma's Revenge, Breakout, and CartPole.

## Setup Instructions

### Prerequisites

- Python 3.7+
- `pip` package manager

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/RL_w_reward_shaping.git
   cd RL_w_reward_shaping
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Training

To train the DQN agent, run the following command:

```bash
python train.py --env <ENV_NAME> --n_episodes <NUM_EPISODES> --target_update_freq <TARGET_UPDATE_FREQ> --render --render_freq <RENDER_FREQ> --save_freq <SAVE_FREQ> --extrinsic
```

Replace <ENV_NAME>, <NUM_EPISODES>, <TARGET_UPDATE_FREQ>, <RENDER_FREQ>, and <SAVE_FREQ> with appropriate values. For example, to train on Montezuma's Revenge:

```bash
python train.py --env MontezumaRevenge-v4 --n_episodes 500 --target_update_freq 100 --render --render_freq 10 --save_freq 50 --extrinsic --ext
```

Parsing Logs and Plotting Results
To parse the log file and plot the training rewards:

```bash
python parse_log.py
```

# Results

## Montezuma's Revenge

### Montezuma's Revenge Results

## Breakout

### Breakout Results

## CartPole

### CartPole Results

## YouTube Video

[Watch the training process on YouTube](insert-your-video-link-here)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
