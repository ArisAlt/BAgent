# src/roi_overlay.py
# version: 0.1.3

import os
import sys
import cv2
import yaml
import numpy as np
from PIL import ImageGrab

# Ensure src/ is on path
BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import capture_utils

class RegionHandler:
    """
    Loads regions from regions.yaml for overlay display.
    """
    YAML_FILENAME = "regions.yaml"

    def __init__(self, yaml_path=None):
        self.yaml_path = yaml_path or os.path.join(BASE_DIR, self.YAML_FILENAME)
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        self.regions = data.get("regions", {})

    def items(self):
        """Yield (name, entry) pairs for all defined regions."""
        for name, entry in self.regions.items():
            yield name, entry

def overlay_regions():
    """
    Capture the EVE client window once, overlay all ROIs as green boxes with labels,
    and wait for ESC to close.
    """
    # Capture window image (auto-detect or full-screen fallback)
    window_img = capture_utils.capture_screen(select_region=False)

    # Retrieve the up-to-date bbox from capture_utils
    bbox = capture_utils._window_bbox
    if bbox is None:
        print("Error: window bounding box not set.")
        return
    x_off, y_off, _, _ = bbox

    # Load defined regions
    rh = RegionHandler()

    # Draw each region
    for name, entry in rh.items():
        rx1, ry1, rx2, ry2 = entry['coords']
        # Convert to window-local coords
        x1 = int(rx1 - x_off)
        y1 = int(ry1 - y_off)
        x2_loc = int(rx2 - x_off)
        y2_loc = int(ry2 - y_off)
        cv2.rectangle(window_img, (x1, y1), (x2_loc, y2_loc), (0, 255, 0), 2)
        cv2.putText(window_img, name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display until ESC pressed
    window_name = "ROI Overlay - Press ESC to exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow(window_name, window_img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    overlay_regions()
