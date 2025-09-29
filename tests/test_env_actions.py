# version: 0.2.0
# path: tests/test_env_actions.py
import os
import sys
import types
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_actions_consistency(monkeypatch):
    # Patch external dependencies used by EveEnv and its submodules
    pg_mod = types.ModuleType('pyautogui')
    pg_mod.moveTo = lambda *a, **kw: None
    pg_mod.click = lambda *a, **kw: None
    pg_mod.keyDown = lambda *a, **kw: None
    pg_mod.keyUp = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, 'pyautogui', pg_mod)

    monkeypatch.setitem(sys.modules, 'cv2', types.ModuleType('cv2'))
    monkeypatch.setitem(sys.modules, 'win32gui', types.ModuleType('win32gui'))

    cap_mod = types.ModuleType('capture_utils')
    cap_mod.capture_screen = lambda select_region=False: np.zeros((10, 10, 3), dtype=np.uint8) if not select_region else (0,0,5,5)
    monkeypatch.setitem(sys.modules, 'capture_utils', cap_mod)

    ocr_mod = types.ModuleType('ocr')
    class DummyOcrEngine:
        def extract_text(self, img):
            return ""
    ocr_mod.OcrEngine = DummyOcrEngine
    monkeypatch.setitem(sys.modules, 'ocr', ocr_mod)

    cv_mod = types.ModuleType('cv')
    class DummyCvEngine:
        def detect_elements(self, *args, **kwargs):
            return []
    cv_mod.CvEngine = DummyCvEngine
    monkeypatch.setitem(sys.modules, 'cv', cv_mod)

    det_mod = types.ModuleType('detector')
    det_mod.load_detector_settings = lambda config_data=None: {}
    det_mod.map_roi_labels = lambda names, entries, cfg=None: {name: {"labels": []} for name in names}
    monkeypatch.setitem(sys.modules, 'detector', det_mod)

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

    rh_mod = types.ModuleType('roi_capture')
    class DummyRegionHandler:
        YAML_FILENAME = 'regions.yaml'
        def __init__(self, yaml_path=None):
            self.yaml_path = yaml_path
            self.regions = {}
        def list_regions(self):
            return []
        def get_type(self, name):
            return 'click'
        def get_coords(self, name):
            return (0,0,1,1)
        def load(self, name):
            return (0,0,1,1)
        def get_screen_resolution(self):
            return (10, 10)
        def validate(self, coords, save_preview=False, region_name='region'):
            return True
    rh_mod.RegionHandler = DummyRegionHandler
    monkeypatch.setitem(sys.modules, 'roi_capture', rh_mod)

    from env import EveEnv

    env = EveEnv()
    assert len(env.actions) == env.action_space.n
    assert len(env.actions) > 0
    first = env.actions[0]
    assert isinstance(first, tuple) and len(first) == 2
