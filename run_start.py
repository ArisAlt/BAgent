# version: 0.2.0
# path: run_start.py

import argparse
from src.env import EveEnv
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser(description="Start EVE Bot Training or Testing")
    parser.add_argument('--train', action='store_true', help='Train the bot from scratch')
    parser.add_argument('--timesteps', type=int, default=10000, help='Number of training timesteps')
    parser.add_argument('--model_path', type=str, default='eve_bot_model', help='Path to save/load the model')
    parser.add_argument('--test', action='store_true', help='Run a test episode')
    args = parser.parse_args()

    env = EveEnv(max_actions=10)

    if args.train:
        model = PPO("MlpPolicy", env, verbose=1)
        print("Starting training loop...")
        model.learn(total_timesteps=args.timesteps)
        model.save(args.model_path)
        print(f"Training complete. Model saved to {args.model_path}")

    if args.test:
        model = PPO.load(args.model_path, env=env)
        print("Running a test episode...")
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Done: {done}")

if __name__ == "__main__":
    main()
