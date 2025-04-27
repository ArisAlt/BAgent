# version: 0.1.1
# path: test_env.py

import random
from src.roi_capture import capture_region_tool, RegionHandler
from src.env import EveEnv


def test_roi_capture():
    print("--- ROI Capture Tool Test ---")
    print("Define or verify your regions for click/text/detect.")
    capture_region_tool()
    regions = RegionHandler().list_regions()
    print("Regions defined:")
    for r in regions:
        print(f" - {r}")


def test_env_steps(num_steps=5):
    print("--- Environment Step Test ---")
    env = EveEnv()
    obs = env.reset()
    print("Initial observation snippet:", obs[:10], "...")

    for i in range(num_steps):
        action = random.randrange(env.action_space.n)
        cmd = env._action_to_command(action)
        print(f"Step {i+1}: Action {action} => Command: {cmd}")
        obs, reward, done, _ = env.step(action)
        print(f"Observation snippet: {obs[:5]}..., Reward: {reward}, Done: {done}\n")
        if done:
            print("Episode done. Resetting environment.")
            obs = env.reset()


if __name__ == "__main__":
    test_roi_capture()
    test_env_steps()
