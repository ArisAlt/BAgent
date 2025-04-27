# version: 0.1.8
# path: src/roi_capture.py

import yaml
import os
import shutil
import cv2
from src.capture_utils import capture_screen

class RegionHandler:
    def __init__(self, yaml_path='regions.yaml', backup_dir='regions_backup', preview_dir='previews'):
        self.yaml_path = yaml_path
        self.backup_dir = backup_dir
        self.preview_dir = preview_dir
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)
        if not os.path.exists(self.yaml_path):
            with open(self.yaml_path, 'w') as f:
                yaml.safe_dump({}, f)

    def load(self, name):
        regions = self._load_all()
        return regions.get(name)

    def get_coords(self, name):
        region = self.load(name)
        if region:
            return (region['x'], region['y'], region['x'] + region['width'], region['y'] + region['height'])
        return None

    def get_type(self, name):
        region = self.load(name)
        return region.get('type') if region else None

    def _load_all(self):
        with open(self.yaml_path) as f:
            return yaml.safe_load(f) or {}

    def list_regions(self):
        return list(self._load_all().keys())

    def delete_region(self, name):
        regions = self._load_all()
        if name in regions:
            self._backup()
            del regions[name]
            with open(self.yaml_path, 'w') as f:
                yaml.safe_dump(regions, f)
            print(f"Deleted region '{name}'")
        else:
            print(f"Region '{name}' not found")

    def save_region(self, name, coords, rtype):
        regions = self._load_all()
        if not name.isidentifier():
            print("Invalid name. Use alphanumeric and underscores only.")
            return
        regions[name] = {
            'x': coords[0], 'y': coords[1],
            'width': coords[2] - coords[0], 'height': coords[3] - coords[1],
            'type': rtype
        }
        self._backup()
        with open(self.yaml_path, 'w') as f:
            yaml.safe_dump(regions, f, default_flow_style=False)
        print(f"Saved region '{name}' ({rtype}): {coords}")

    def _backup(self):
        if os.path.exists(self.yaml_path):
            shutil.copy(self.yaml_path, os.path.join(self.backup_dir, os.path.basename(self.yaml_path)))

    def preview(self, name):
        region = self.load(name)
        if not region:
            print(f"Region '{name}' not defined")
            return
        coords = self.get_coords(name)
        img = capture_screen()
        x1, y1, x2, y2 = coords
        roi = img[y1:y2, x1:x2]
        preview_path = os.path.join(self.preview_dir, f"{name}.png")
        cv2.imwrite(preview_path, roi)
        print(f"Preview saved to {preview_path}")


def capture_region_tool():
    handler = RegionHandler()
    recommended = [
        'mining_lasers', 'cargo_hold', 'hostile_warning',
        'station_warp', 'dock_button', 'undock_button', 'overview_panel',
        'asteroid_field', 'asteroid_entry', 'approach_button',
        'shield_status', 'armor_status', 'capacitor_status',
        'module_slot1_status', 'module_slot2_status', 'module_slot3_status'
    ]
    print("=== ROI Capture Tool ===")
    print("Recommended regions:")
    for name in recommended:
        print(f"  - {name}")
    print("Commands:")
    print("  capture <region_name>")
    print("  list")
    print("  delete <region_name>")
    print("  preview <region_name>")
    print("  exit")
    print("Note: capture will prompt for region type (click/text/detect)")

    while True:
        cmd = input("roi> ").strip().split()
        if not cmd:
            continue
        action = cmd[0].lower()
        if action == 'capture' and len(cmd) == 2:
            name = cmd[1]
            rtype = input("Enter region type (click/text/detect): ").strip().lower()
            if rtype not in ('click', 'text', 'detect'):
                print("Invalid type. Choose click, text, or detect.")
                continue
            print(f"Select region for '{name}' ({rtype}), then press ENTER...")
            coords = capture_screen(select_region=True)
            if coords:
                handler.save_region(name, coords, rtype)
        elif action == 'list':
            for r in handler.list_regions():
                print(f" - {r}")
        elif action == 'delete' and len(cmd) == 2:
            handler.delete_region(cmd[1])
        elif action == 'preview' and len(cmd) == 2:
            handler.preview(cmd[1])
        elif action == 'exit':
            break
        else:
            print("Unknown command")

if __name__ == '__main__':
    capture_region_tool()
