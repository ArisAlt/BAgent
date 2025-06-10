# version: 0.3.0
# path: data_recorder.py

import pickle
import random
import os
from src.env import EveEnv
from stable_baselines3 import PPO


def record_data(filename='demo_buffer.pkl', num_samples=500, manual=True, model_path=None):
    env = EveEnv()
    demo_buffer = []
    model = None
    if model_path and os.path.exists(model_path):
        model = PPO.load(model_path, env=env)

    mode = "manual" if manual else ("model" if model else "automatic")
    print(f"Starting {mode} data recording for {num_samples} samples...")
    obs = env.reset()

    for i in range(num_samples):
        if manual:
            action = int(input(f"Step {i+1}/{num_samples} - Enter action (0 to {env.action_space.n - 1}): "))
        elif model:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = random.randint(0, env.action_space.n - 1)
        obs, reward, done, info = env.step(action)
        demo_buffer.append((obs, action))
        print(f"Recorded: Step={i+1}, Action={action}, Reward={reward}")
        if done:
            obs = env.reset()

    with open(filename, 'wb') as f:
        pickle.dump(demo_buffer, f)

    print(f"Data recording complete. Saved to {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record environment actions")
    parser.add_argument("--out", type=str, default="demo_buffer.pkl")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    record_data(filename=args.out, num_samples=args.samples,
                manual=args.manual, model_path=args.model)
  