# version: 0.1.6
# path: src/roi_capture.py

import pyautogui
import yaml
import os
from PIL import ImageGrab

class RegionHandler:
    def load(self, region_name, yaml_path='regions.yaml'):
        with open(yaml_path, 'r') as f:
            regions = yaml.safe_load(f)
        region = regions.get(region_name)
        if region:
            loaded_region = (region['x'], region['y'], region['x'] + region['width'], region['y'] + region['height'])
            print(f"Loaded region '{region_name}': {loaded_region}")
            return loaded_region
        else:
            print(f"Region '{region_name}' not found in {yaml_path}.")
            return None

    def validate(self, capture_region, save_preview=False, region_name=None, preview_dir='previews'):
        if capture_region:
            img = ImageGrab.grab(bbox=capture_region)
            img.show()
            print(f"Displayed region {capture_region} for validation.")
            if save_preview and region_name:
                os.makedirs(preview_dir, exist_ok=True)
                preview_path = os.path.join(preview_dir, f"{region_name}.png")
                img.save(preview_path)
                print(f"Preview saved to {preview_path}")
        else:
            print("No capture region set for validation.")

    def list_regions(self, yaml_path='regions.yaml'):
        with open(yaml_path, 'r') as f:
            regions = yaml.safe_load(f)
        print("Available Regions:")
        for name in regions:
            print(f" - {name}: {regions[name]}")

    def delete_region(self, region_name, yaml_path='regions.yaml'):
        with open(yaml_path, 'r') as f:
            regions = yaml.safe_load(f)
        if region_name in regions:
            del regions[region_name]
            with open(yaml_path, 'w') as f:
                yaml.safe_dump(regions, f)
            print(f"Region '{region_name}' deleted from {yaml_path}.")
        else:
            print(f"Region '{region_name}' not found in {yaml_path}.")

def capture_region_tool(save_path='regions.yaml'):
    regions = {}
    print("================= ROI CAPTURE TOOL =================")
    print("Instructions:")
    print("1. Move your mouse to the top-left corner of the desired region.")
    print("2. Press Enter to record the top-left corner.")
    print("3. Move your mouse to the bottom-right corner of the desired region.")
    print("4. Press Enter to record the bottom-right corner.")
    print("5. Enter a name for the region (alphanumeric, underscores allowed).")
    print("6. Recommended Regions to Capture:")
    print("   - overview_panel")
    print("   - mining_lasers")
    print("   - cargo_hold")
    print("   - system_status")
    print("   - hostile_warning")
    print("   - chat_window")
    print("7. Repeat or quit when done.")
    print("====================================================")

    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            regions = yaml.safe_load(f) or {}

    while True:
        print(f"Move to top-left of region and press Enter")
        input()
        x1, y1 = pyautogui.position()
        print(f"Top-left at: ({x1}, {y1})")
        print(f"Move to bottom-right of region and press Enter")
        input()
        x2, y2 = pyautogui.position()
        print(f"Bottom-right at: ({x2}, {y2})")
        if x2 <= x1 or y2 <= y1:
            print("Invalid coordinates: bottom-right must be greater than top-left. Try again.")
            continue
        name = input("Enter region name: ")
        if not name.isidentifier():
            print("Invalid name: must be alphanumeric with optional underscores. Try again.")
            continue
        if name in regions:
            print(f"Region '{name}' already exists and will be overwritten.")
        regions[name] = {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}
        cont = input("Capture another region? (y/n): ")
        if cont.lower() != 'y':
            break

    if os.path.exists(save_path):
        backup_path = save_path + '.bak'
        os.rename(save_path, backup_path)
        print(f"Backup of previous regions saved to {backup_path}")

    with open(save_path, 'w') as f:
        yaml.safe_dump(regions, f, default_flow_style=False)
    print(f"Regions saved to {save_path}")
