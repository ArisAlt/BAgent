import sys
import importlib
import types
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Prepare dummy modules to avoid real GUI dependencies
cv2_mod = types.ModuleType('cv2')
cv2_called = {}

def fake_selectROI(winname, img, showCrosshair=True, fromCenter=False):
    cv2_called['img_shape'] = img.shape
    return (1, 2, 3, 4)

cv2_mod.selectROI = fake_selectROI
cv2_mod.destroyWindow = lambda name: None

sys.modules['cv2'] = cv2_mod
sys.modules['win32gui'] = None
sys.modules['win32ui'] = None
sys.modules['win32con'] = None

if 'capture_utils' in sys.modules:
    del sys.modules['capture_utils']
import capture_utils
importlib.reload(capture_utils)

def test_select_region_returns_coords():
    coords = capture_utils.capture_screen(select_region=True)
    assert coords == (1, 2, 4, 6)
    assert cv2_called['img_shape'] == (10, 10, 3)

