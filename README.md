
# RL-agent-for-Maintenance-scheduling-of-bridges

This repository contains a reinforcement learning (RL) framework for autonomous maintenance scheduling of bridges, prioritizing both the age of bridges and the capital involved in renovations.

## Deterioration Model
The deterioration model is implemented in `deterioration_model.py` as a custom Gymnasium environment. Key features:

- **State Space:** Each bridge is represented by features such as traffic intensity, location, age, auxiliary age, failure probability, survival function, and capital invested.
- **Action Space:** The agent can choose from actions like doing nothing, monitoring, minor/medium/major intervention, or replacement for each bridge.
- **Transition Dynamics:** Actions affect the bridge's auxiliary age, failure probability, and capital invested. The survival function is updated using a Weibull model.
- **Reward Function:** The agent receives a negative reward (cost) based on the sum of auxiliary ages, failure probabilities, and intervention costs, encouraging it to minimize both risk and expense.

## Training Details
The training process uses the PPO algorithm from Stable Baselines3. The environment is vectorized and normalized for stable learning. Training details:

- **Algorithm:** PPO (Proximal Policy Optimization)
- **Policy:** MLP with two hidden layers of 128 units each, ReLU activation
- **Training Steps:** 100,000
- **Environment Normalization:** Observations are normalized using `VecNormalize`
- **Evaluation:** After training, the agent is evaluated over 10 episodes to report average cost/reward

### How to Train
1. Install dependencies:
	```
	pip install -r requirements.txt
	pip install stable-baselines3 torch
	```
2. Run the training script:
	```
	python training.py
	```
3. The trained model and normalization statistics will be saved in the `models/` directory.

