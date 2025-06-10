import os
import sys
import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Skip entire module if required deps missing
pytest.importorskip('gym')
pytest.importorskip('pyautogui')
pytest.importorskip('cv2')
pytest.importorskip('pytesseract')

from env import EveEnv

def test_env_reset_and_step():
    env = EveEnv()
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape

    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)
