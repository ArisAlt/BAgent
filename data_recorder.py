# version: 0.3.0
# path: data_recorder.py

import pickle
import random
from src.env import EveEnv


def record_data(filename='demo_buffer.pkl', num_samples=500, manual=True):
    env = EveEnv()
    demo_buffer = []

    mode = "manual" if manual else "automatic"
    print(f"Starting {mode} data recording for {num_samples} samples...")
    obs = env.reset()

    for i in range(num_samples):
        if manual:
            action = int(input(f"Step {i+1}/{num_samples} - Enter action (0 to {env.action_space.n - 1}): "))
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
    # Change manual to False later for automatic mode
    record_data(manual=False)
  