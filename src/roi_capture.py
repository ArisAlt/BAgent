# version: 0.2.4
# path: src/roi_capture.py

import os
import sys
import cv2
import yaml
import numpy as np
from datetime import datetime
from capture_utils import capture_screen

BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

class RegionHandler:
    YAML_FILENAME = "regions.yaml"
    PREVIEW_DIR = "previews"
    SCREENSHOT_DIR = "roi_screenshots"

    def __init__(self, yaml_path=None):
        self.yaml_path = yaml_path or os.path.join(BASE_DIR, self.YAML_FILENAME)
        self.current_resolution = self.get_screen_resolution()
        self._load()

    def get_screen_resolution(self):
        img = capture_screen(select_region=False)
        return img.shape[1], img.shape[0]

    def _load(self):
        if os.path.exists(self.yaml_path):
            with open(self.yaml_path,'r') as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        self.regions = data.get('regions', {})

    def _backup(self):
        if os.path.exists(self.yaml_path):
            os.replace(self.yaml_path, self.yaml_path + '.bak')

    def _save(self):
        os.makedirs(os.path.dirname(self.yaml_path), exist_ok=True)
        with open(self.yaml_path,'w') as f:
            yaml.safe_dump({'regions': self.regions}, f)

    def list_regions(self):
        return list(self.regions.keys())

    def get_type(self, name):
        """Return the stored type for a region or None."""
        entry = self.regions.get(name)
        if entry:
            return entry.get('type')
        return None

    def get_coords(self, name):
        """Convenience wrapper around :meth:`load`."""
        return self.load(name)

    def add_region(self, name, abs_coords, region_type):
        if not name.isidentifier():
            raise ValueError("Invalid region name.")
        x1,y1,x2,y2 = abs_coords
        if not (0<=x1<x2 and 0<=y1<y2):
            raise ValueError("Coordinates out of order or bounds.")
        w,h = self.current_resolution
        rel = [x1/w, y1/h, x2/w, y2/h]
        self._backup()
        self.regions[name] = {'rel': rel, 'type': region_type}
        self._save()

    def delete_region(self, name):
        if name in self.regions:
            self._backup()
            del self.regions[name]
            self._save()

    def load(self, name):
        entry = self.regions.get(name)
        if not entry:
            return None
        rel = entry.get('rel')
        if isinstance(rel, (list,tuple)) and len(rel)==4:
            w,h = self.get_screen_resolution()
            x1 = int(rel[0]*w); y1 = int(rel[1]*h)
            x2 = int(rel[2]*w); y2 = int(rel[3]*h)
            return (x1,y1,x2,y2)
        coords = entry.get('coords')
        if isinstance(coords, (list,tuple)) and len(coords)==4:
            return tuple(coords)
        return None

    def preview_region(self, name):
        coords = self.load(name)
        if not coords:
            return False
        x1,y1,x2,y2 = coords
        img = capture_screen(select_region=False)
        crop = img[y1:y2, x1:x2]
        preview_dir = os.path.join(BASE_DIR, self.PREVIEW_DIR)
        os.makedirs(preview_dir, exist_ok=True)
        out = os.path.join(preview_dir, f"{name}.png")
        cv2.imwrite(out, crop)
        cv2.imshow(f"Preview: {name}", crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True

    def validate(self, coords, save_preview=False, region_name="region"):
        """Validate coordinates fall inside the current screen resolution."""
        if not coords:
            return False
        x1, y1, x2, y2 = coords
        w, h = self.get_screen_resolution()
        valid = 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        if valid and save_preview:
            img = capture_screen(select_region=False)
            crop = img[y1:y2, x1:x2]
            sd = os.path.join(BASE_DIR, self.PREVIEW_DIR)
            os.makedirs(sd, exist_ok=True)
            fp = os.path.join(sd, f"{region_name}_preview.png")
            cv2.imwrite(fp, crop)
        return valid


def print_menu():
    print("\nROI Capture Tool Commands:")
    print("  capture <name>   - Define or update a region")
    print("  list             - List all defined regions")
    print("  delete <name>    - Delete a region")
    print("  preview <name>   - Show and save region preview")
    print("  help             - Show this menu again")
    print("  exit, quit       - Exit tool")

    # Updated recommended regions including new YAML-based ROIs
    recommended = [
        # Core UI panels
        "overview_panel", "selected_item_window", "module_rack",
        # Ship stats
        "shield_status", "armor_status", "capacitor_status", "cargo_hold_status",
        # Actions
        "dock_button", "undock_button", "warp_button", "approach_button",
        # Modules slots
        "module_slot1", "module_slot2", "module_slot3",
        # Drones and targets
        "drone_window", "hostile_warning", "targeting_indicators",
        # Notifications and alerts
        "alerts_notifications",
        # Mining-specific
        "asteroi_dfield", "asteroid_entry", "mining_lasers","loacation_panel",
        "overview_tab_mining",  # if you capture the tab selector
        # Cargo operations
        "move_to_station",
        # Mission and exploration
        "scan_button", "anomaly_list", "accept_mission_button", "missions_tab",
        # Locations panel and bookmarks
        "locations_window"
    ]
    print("\nRecommended Regions (from config):")
    for r in recommended:
        print(f"  • {r}")

    # Show any existing regions
    rh = RegionHandler()
    regs = rh.list_regions()
    if regs:
        print("\nCurrent Defined ROIs:")
        for n in regs:
            coords = rh.load(n)
            rtype = rh.regions.get(n, {}).get('type', 'unknown')
            print(f"  • {n}: {coords} (type: {rtype})")
    else:
        print("\nNo ROIs defined yet.")

def capture_region_tool():
    rh = RegionHandler()
    print("ROI Capture Tool started. Type 'help' for commands.")
    print_menu()
    while True:
        cmd = input("roi> ").strip().split()
        if not cmd:
            continue
        op = cmd[0]

        if op == 'help':
            print_menu()

        elif op == 'capture' and len(cmd)==2:
            name = cmd[1]
            rt = input("Enter region type (click/text/detect): ").strip()
            print(f"Select region '{name}' then press ENTER...")
            abs_coords = capture_screen(select_region=True)
            if abs_coords is None:
                print("Cancelled.")
                continue
            if isinstance(abs_coords, np.ndarray):
                abs_coords = tuple(abs_coords.flatten().astype(int))
            rh.add_region(name, abs_coords, rt)
            # Save screenshot of captured region with configurable padding
            coords = rh.load(name)
            img = capture_screen(select_region=False)
            x1, y1, x2, y2 = coords
            # Add padding to left and right to ensure full capture
            pad_left = 5   # pixels to shift right from original x1
            pad_right = 5  # pixels to expand on right
            h_img, w_img = img.shape[:2]
            # Shift start and expand end
            x1_padded = min(w_img, x1 + pad_left)
            x2_padded = min(w_img, x2 + pad_right)
            crop = img[y1:y2, x1_padded:x2_padded]
            sd = os.path.join(BASE_DIR, RegionHandler.SCREENSHOT_DIR)
            os.makedirs(sd, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fp = os.path.join(sd, f"{name}_{ts}.png")
            cv2.imwrite(fp, crop)
            print(f"Region '{name}' captured. Screenshot saved: {fp}")

        elif op == 'list':
            # List regions with absolute coordinates and types
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

if __name__=="__main__":
    capture_region_tool()
