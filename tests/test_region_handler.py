import os
import sys
import yaml
import pytest
import types

# Provide a dummy cv2 module if OpenCV is not installed
if 'cv2' not in sys.modules:
    sys.modules['cv2'] = types.ModuleType('cv2')

if 'PIL' not in sys.modules:
    pil_mod = types.ModuleType('PIL')
    grab_mod = types.ModuleType('PIL.ImageGrab')
    grab_mod.grab = lambda *a, **kw: None
    pil_mod.ImageGrab = grab_mod
    sys.modules['PIL'] = pil_mod
    sys.modules['PIL.ImageGrab'] = grab_mod

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from roi_capture import RegionHandler


def test_load_and_type(tmp_path, monkeypatch):
    data = {
        'regions': {
            'sample': {
                'rel': [0.1, 0.2, 0.3, 0.4],
                'type': 'click'
            }
        }
    }
    yaml_file = tmp_path / 'regions.yaml'
    yaml_file.write_text(yaml.safe_dump(data))

    monkeypatch.setattr(RegionHandler, 'get_screen_resolution', lambda self: (200, 100))

    rh = RegionHandler(yaml_path=str(yaml_file))
    coords = rh.load('sample')
    assert coords == (20, 20, 60, 40)

    if hasattr(rh, 'get_type'):
        assert rh.get_type('sample') == 'click'
    else:
        assert rh.regions['sample']['type'] == 'click'
