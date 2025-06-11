# roi_capture.py
# version: 0.1.2
# path: roi_capture.py
"""Thin wrapper for :mod:`src.roi_capture` with graceful fallback.

Provides basic stubs for ``cv2`` so that unit tests can run without the
real OpenCV dependency installed.  The stubs implement just enough
functionality for ``capture_utils`` and ``roi_capture`` to import
successfully.
"""

import importlib, types, sys

if 'cv2' not in sys.modules:
    cv2_stub = types.ModuleType('cv2')
    cv2_stub.imwrite = lambda *a, **k: None
    cv2_stub.imshow = lambda *a, **k: None
    cv2_stub.waitKey = lambda *a, **k: None
    cv2_stub.destroyAllWindows = lambda *a, **k: None
    cv2_stub.cvtColor = lambda img, flag=None: img
    cv2_stub.COLOR_BGRA2BGR = 0
    sys.modules['cv2'] = cv2_stub

try:
    module = importlib.import_module('src.roi_capture')
    globals().update(module.__dict__)

    # Minimal wrapper so tests can override RegionHandler and capture_screen
    def capture_region_tool():
        RH = globals().get('RegionHandler', module.RegionHandler)
        cs = globals().get('capture_screen', module.capture_screen)
        rh = RH()
        print("ROI Capture Tool started. Type 'help' for commands.")
        module.print_menu()
        while True:
            cmd = input("roi> ").strip().split()
            if not cmd:
                continue
            op = cmd[0]
            if op == 'help':
                module.print_menu()
            elif op == 'capture' and len(cmd)==2:
                name = cmd[1]
                rt = input("Enter region type (click/text/detect): ").strip()
                print(f"Select region '{name}' then press ENTER...")
                abs_coords = cs(select_region=True)
                if abs_coords is None:
                    print("Cancelled.")
                    continue
                if isinstance(abs_coords, np.ndarray):
                    abs_coords = tuple(abs_coords.flatten().astype(int))
                rh.add_region(name, abs_coords, rt)
                print(f"Region '{name}' captured.")
            elif op == 'list':
                regs = rh.list_regions()
                if not regs:
                    print("No regions defined.")
                else:
                    print("Defined regions:")
                    for n in regs:
                        coords = rh.load(n)
                        rtype = rh.regions.get(n, {}).get('type', 'unknown')
                        print(f"  • {n}: {coords} (type: {rtype})")
            elif op == 'delete' and len(cmd)==2:
                rh.delete_region(cmd[1])
                print(f"Region '{cmd[1]}' deleted.")
            elif op == 'preview' and len(cmd)==2:
                if not rh.preview_region(cmd[1]):
                    print("Region not found.")
            elif op == 'exit':
                break
            else:
                print("Unknown command. Type 'help' for commands.")
    RegionHandler = module.RegionHandler
except Exception:  # pragma: no cover - dependencies missing
    def capture_region_tool(*args, **kwargs):
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
