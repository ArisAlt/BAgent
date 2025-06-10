import os
import sys
import types
import random
import unittest

# Provide minimal stub implementations for required external modules

# ---- numpy stub ----
class _FakeArray(list):
    @property
    def shape(self):
        return (len(self),)

def _np_array(data, dtype=None):
    return _FakeArray(data)

def _np_zeros(shape, dtype=None):
    return _FakeArray([0] * shape[0])

fake_np = types.ModuleType('numpy')
fake_np.array = _np_array
fake_np.zeros = _np_zeros
fake_np.inf = float('inf')
fake_np.float32 = float
sys.modules['numpy'] = fake_np

# ---- gym stub ----
class _FakeEnv:
    pass

class _FakeDiscrete:
    def __init__(self, n):
        self.n = n
    def sample(self):
        return random.randrange(self.n)

class _FakeBox:
    def __init__(self, low, high, shape, dtype=None):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

fake_spaces = types.SimpleNamespace(Discrete=_FakeDiscrete, Box=_FakeBox)
fake_gym = types.ModuleType('gym')
fake_gym.Env = _FakeEnv
fake_gym.spaces = fake_spaces
sys.modules['gym'] = fake_gym
sys.modules['gym.spaces'] = fake_spaces

# ---- stubs for project modules ----
class DummyOcr:
    def extract_text(self, img):
        return "0/1"

class DummyCv:
    def detect_elements(self, img, templates=None):
        return []

class DummyUi:
    def capture(self):
        class Img:
            def __getitem__(self, key):
                return self
        return Img()
    def execute(self, cmd):
        pass

class DummyRegionHandler:
    def list_regions(self):
        return []
    def get_type(self, name):
        return 'click'
    def get_coords(self, name):
        return None

sys.modules['ocr'] = types.SimpleNamespace(OcrEngine=DummyOcr)
sys.modules['cv'] = types.SimpleNamespace(CvEngine=DummyCv)
sys.modules['ui'] = types.SimpleNamespace(Ui=DummyUi)
sys.modules['roi_capture'] = types.SimpleNamespace(RegionHandler=DummyRegionHandler)

# Ensure src/ is on the path and import EveEnv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from env import EveEnv


class EveEnvTests(unittest.TestCase):
    def setUp(self):
        self.env = EveEnv()

    def test_reset_shape(self):
        obs = self.env.reset()
        self.assertEqual(len(obs), self.env.observation_space.shape[0])

    def test_step_returns_types(self):
        self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, done, info = self.env.step(action)
        self.assertIsInstance(obs, list)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)


if __name__ == '__main__':
    unittest.main()
