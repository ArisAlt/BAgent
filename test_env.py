# test_env.py
# Quick smoke-test for your ROI definitions and Gym wrapper

import time
import random
import pytest

try:
    import cv2  # noqa: F401
except Exception:  # pragma: no cover - skip if cv2 unavailable
    pytest.skip("cv2 not available", allow_module_level=True)

# Make sure Python can import your modules
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from roi_capture import RegionHandler
from env import EveEnv

def test_rois():
    """List out all ROIs you’ve previously captured."""
    rh = RegionHandler()
    regions = rh.list_regions()
    if not regions:
        print("⚠️ No ROIs defined yet. Run your ROI tool first to capture regions.")
    else:
        print("✅ Defined ROIs:")
        for name in regions:
            print(f"  • {name}")

def test_env_steps(num_steps: int = 5):
    """Reset the environment and take a few random actions."""
    env = EveEnv()
    assert len(env.actions) == env.action_space.n
    assert len(env.actions) > 0
    assert isinstance(env.actions[0], tuple) and len(env.actions[0]) == 2
    obs = env.reset()
    print(f"\n[Env] Reset → initial obs snippet: {obs[:5]}…\n")

    for i in range(num_steps):
        action = random.randrange(env.action_space.n)
        cmd = env._action_to_command(action)
        obs, reward, done, _ = env.step(action)
        print(f"Step {i+1}/{num_steps}")
        print(f"  Action idx: {action}")
        print(f"  Mapped cmd: {cmd}")
        print(f"  Reward:      {reward:.2f}")
        print(f"  Done?        {done}")
        print(f"  Obs snippet: {obs[:5]}…\n")
        if done:
            print("▶️ Episode ended early.")
            break
        time.sleep(0.5)

if __name__ == "__main__":
    print("=== ROI Definitions Check ===")
    test_rois()
    print("\n=== Environment Step Test ===")
    test_env_steps()
