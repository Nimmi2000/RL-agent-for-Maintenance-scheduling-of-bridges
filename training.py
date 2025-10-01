import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from deterioration_model import bridgedeterioration
import torch as th

def main():
    # Path to bridge data
    data_path = 'bridge_data.csv'
    # Create environment
    env = bridgedeterioration(df_path=data_path)
    env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=float('inf'))

    # PPO policy arguments
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128])
    norm_env.reset()
    model = PPO(
        "MlpPolicy",
        norm_env,
        n_epochs=10,
        learning_rate=0.0003,
        batch_size=64,
        ent_coef=0.01,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        verbose=2
    )
    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=100000, log_interval=10)

    # Save model and normalization stats
    os.makedirs('models', exist_ok=True)
    model.save('models/ppo_bridge_maintenance')
    norm_env.save('models/vecnormalize.pkl')
    print('Training complete. Model and normalization stats saved in models/.')

    # --- Evaluation ---
    print("\nEvaluating trained model...")
    norm_env.training = False
    obs = norm_env.reset()
    total_rewards = []
    num_episodes = 10
    max_steps = 50
    for ep in range(num_episodes):
        ep_reward = 0
        obs = norm_env.reset()
        done = [False]
        steps = 0
        while not all(done) and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = norm_env.step(action)
            ep_reward += reward[0]
            steps += 1
        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: Total Reward (Cost): {ep_reward:.2f}")
    avg_reward = sum(total_rewards) / num_episodes
    print(f"\nAverage Reward (Cost) over {num_episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    main()
