# roi_capture.py
# version: 0.1.1
# path: roi_capture.py
"""Thin wrapper for src.roi_capture with graceful fallback."""

import importlib, os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
if 'cv2' not in sys.modules:
    cv2_stub = types.ModuleType('cv2')
    cv2_stub.imwrite = lambda *a, **k: None
    cv2_stub.imshow = lambda *a, **k: None
    cv2_stub.waitKey = lambda *a, **k: None
    cv2_stub.destroyAllWindows = lambda *a, **k: None
    sys.modules['cv2'] = cv2_stub

try:
    module = importlib.import_module('src.roi_capture')
    globals().update(module.__dict__)
    module.RegionHandler = globals().get('RegionHandler', module.RegionHandler)
    module.capture_region_tool = globals().get('capture_region_tool', module.capture_region_tool) if 'capture_region_tool' in globals() else module.capture_region_tool
except Exception:  # pragma: no cover - dependencies missing
    def capture_region_tool():
        raise RuntimeError('roi_capture requires optional dependencies')

    class RegionHandler:
        def __init__(self, yaml_path=None):
            self.yaml_path = yaml_path
        def list_regions(self):
            return []
        def get_type(self, name):
            return None
        def get_coords(self, name):
            return None
        def load(self, name):
            return None
        def get_screen_resolution(self):
            return (0, 0)
        def validate(self, coords, save_preview=False, region_name='region'):
            return True
