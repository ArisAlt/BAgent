import os
import sys
import types
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def setup_env(monkeypatch):
    # Dummy capture_utils
    cap_mod = types.ModuleType('capture_utils')
    cap_mod.capture_screen = lambda select_region=False: (
        np.zeros((10, 10, 3), dtype=np.uint8)
        if not select_region else (0, 0, 5, 5)
    )
    monkeypatch.setitem(sys.modules, 'capture_utils', cap_mod)

    # Dummy OCR engine
    ocr_mod = types.ModuleType('ocr')

    class DummyOcrEngine:
        def extract_text(self, img):
            return "10/100"

    ocr_mod.OcrEngine = DummyOcrEngine
    monkeypatch.setitem(sys.modules, 'ocr', ocr_mod)

    # Dummy CV engine
    cv_mod = types.ModuleType('cv')

    class DummyCvEngine:
        def detect_elements(self, img, templates=None, threshold=0.8, multi_scale=False, scales=None):
            if templates and 'mining_laser_on' in templates:
                return [{'name': 'mining_laser_on'}]
            return []

    cv_mod.CvEngine = DummyCvEngine
    monkeypatch.setitem(sys.modules, 'cv', cv_mod)

    # Dummy UI
    ui_mod = types.ModuleType('ui')

    class DummyUi:
        def __init__(self, capture_region=None):
            self.capture_region = capture_region

        def capture(self):
            return np.zeros((10, 10, 3), dtype=np.uint8)

        def execute(self, command):
            pass

    ui_mod.Ui = DummyUi
    monkeypatch.setitem(sys.modules, 'ui', ui_mod)

    # Dummy RegionHandler
    rh_mod = types.ModuleType('roi_capture')

    class DummyRegionHandler:
        YAML_FILENAME = 'regions.yaml'

        def __init__(self, yaml_path=None):
            self.yaml_path = yaml_path

        def list_regions(self):
            return ['roi_click', 'cargo_hold', 'mining_laser_on']

        def get_type(self, name):
            return {
                'roi_click': 'click',
                'cargo_hold': 'text',
                'mining_laser_on': 'detect',
            }.get(name, 'click')

        def get_coords(self, name):
            return (0, 0, 1, 1)

        def load(self, name):
            return (0, 0, 1, 1)

        def get_screen_resolution(self):
            return (10, 10)

        def validate(self, coords, save_preview=False, region_name='region'):
            return True

    rh_mod.RegionHandler = DummyRegionHandler
    monkeypatch.setitem(sys.modules, 'roi_capture', rh_mod)

    return cap_mod, ocr_mod, cv_mod, ui_mod, rh_mod


def test_env_reset_and_step(monkeypatch):
    setup_env(monkeypatch)
    import importlib
    import env
    importlib.reload(env)
    EveEnv = env.EveEnv

    env = EveEnv()
    assert env.action_space.n == 7

    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape

    obs2, reward, done, info = env.step(0)
    assert obs2.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert reward > 0
    assert isinstance(done, bool)
    assert isinstance(info, dict)
