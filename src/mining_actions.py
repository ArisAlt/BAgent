# version: 0.1.2
# path: src/mining_actions.py
"""High level mining routine utilities.

This module implements helper methods corresponding to the
"Recommended Mining Actions" list from the project scaffold.
All functions rely on existing ROI definitions and the ``agent_config.yaml``
file. The implementation favors simple UI automation with randomized
jitter to mimic human behaviour.
"""

import os
import random
import time
import yaml
import pyautogui

from .ui import Ui
from .roi_capture import RegionHandler
from .ocr import OcrEngine
from .cv import CvEngine
from . import capture_utils


class MiningActions:
    """Utility class encapsulating common mining tasks."""

    def __init__(self, ui=None, region_handler=None, ocr=None, cv=None, config_path=None):
        self.ui = ui or Ui()
        self.rh = region_handler or RegionHandler()
        self.ocr = ocr or OcrEngine()
        self.cv = cv or CvEngine()
        cfg_path = config_path or os.path.join(os.path.dirname(__file__), "config", "agent_config.yaml")
        with open(cfg_path, "r") as f:
            self.config = yaml.safe_load(f)
        self._belt_index = 0

    def _sleep(self, base=1.0):
        time.sleep(base * random.uniform(0.85, 1.15))

    # --- Core actions -----------------------------------------------------

    def warp_to_asteroid_belt(self):
        """Warp to the next asteroid belt bookmark."""
        loc_cfg = self.config.get("locations", {})
        belts = loc_cfg.get("belt_bookmarks", [])
        if not belts:
            return
        bookmark = belts[self._belt_index % len(belts)]
        self._belt_index += 1

        window_roi = loc_cfg.get("window_roi")
        coords = self.rh.load(window_roi)
        if not coords:
            return
        x1, y1, x2, y2 = coords
        pyautogui.click((x1 + x2) // 2, (y1 + y2) // 2)
        self._sleep(0.5)

        full = capture_utils.capture_screen(select_region=False)
        panel = full[y1:y2, x1:x2]
        data = self.ocr.extract_data(panel)
        for entry in data:
            if entry["text"].strip() == bookmark:
                bx = x1 + entry["left"]
                by = y1 + entry["top"]
                bw = entry["width"]
                bh = entry["height"]
                pyautogui.click(bx + bw // 2, by + bh // 2)
                break

        warp_roi = self.rh.load("warp_button")
        if warp_roi:
            cx = (warp_roi[0] + warp_roi[2]) // 2
            cy = (warp_roi[1] + warp_roi[3]) // 2
            pyautogui.click(cx, cy)
        self._sleep(2.0)

    def approach_asteroid(self):
        box = self.rh.load("approach_button")
        if box:
            x1, y1, x2, y2 = box
            pyautogui.click((x1 + x2) // 2, (y1 + y2) // 2)
            self._sleep(1.0)

    def target_asteroid_via_ocr(self):
        """Scan the overview for an asteroid entry using OCR and click it."""
        panel = self.rh.load("overview_panel")
        if not panel:
            return False

        x1, y1, x2, y2 = panel
        screen = capture_utils.capture_screen(select_region=False)
        crop = screen[y1:y2, x1:x2]
        pos = self.cv.find_asteroid_entry(crop)
        if pos:
            cx, cy = pos
            pyautogui.click(x1 + cx, y1 + cy)
            self._sleep(0.5)
            return True
        return False

    def activate_mining_lasers(self):
        slots = ["module_slot1", "module_slot2", "module_slot3"]
        for slot in slots:
            box = self.rh.load(slot)
            if not box:
                continue
            x1, y1, x2, y2 = box
            active = self.cv.is_module_active(capture_utils.capture_screen()[y1:y2, x1:x2])
            if not active:
                pyautogui.click((x1 + x2) // 2, (y1 + y2) // 2)
                self._sleep(0.2)
        self._sleep(0.5)

    def monitor_cycle_completion(self):
        # Placeholder using module active detection
        pass

    def check_cargo_hold(self):
        box = self.rh.load("mining_cargo_hold_capacity")
        if not box:
            return 0
        x1, y1, x2, y2 = box
        img = capture_utils.capture_screen(select_region=False)[y1:y2, x1:x2]
        text = self.ocr.extract_text(img)
        numbers = [int(s) for s in text.split() if s.isdigit()]
        return numbers[0] if numbers else 0

    def recover_ore_fragments(self):
        # Placeholder for tractor beam or drone commands
        pass

    def filter_and_jettison_unwanted(self):
        # Placeholder for cargo filtering and jettison logic
        pass

    def randomize_camera_movement(self):
        dx = random.randint(-50, 50)
        dy = random.randint(-50, 50)
        duration = random.uniform(0.2, 0.5)
        pyautogui.moveRel(dx, dy, duration=duration)
        self._sleep(0.2)

    def adjust_overview_filters(self):
        # Placeholder for overview filter adjustments
        pass

    def detect_hostiles(self, screen=None):
        if screen is None:
            screen = capture_utils.capture_screen(select_region=False)
        warn_box = self.rh.load("hostile_warning")
        if not warn_box:
            return False
        x1, y1, x2, y2 = warn_box
        region = screen[y1:y2, x1:x2]
        elements = self.cv.detect_elements(region, templates={"alert": "templates/hostile_alert.png"})
        return bool(elements)

    def warp_to_station(self):
        loc_cfg = self.config.get("locations", {})
        window_roi = loc_cfg.get("window_roi")
        station = loc_cfg.get("station_bookmark")
        coords = self.rh.load(window_roi)
        if not (coords and station):
            return
        x1, y1, x2, y2 = coords
        pyautogui.click((x1 + x2) // 2, (y1 + y2) // 2)
        self._sleep(0.5)

        full = capture_utils.capture_screen(select_region=False)
        panel = full[y1:y2, x1:x2]
        data = self.ocr.extract_data(panel)
        for entry in data:
            if entry["text"].strip() == station:
                bx = x1 + entry["left"]
                by = y1 + entry["top"]
                bw = entry["width"]
                bh = entry["height"]
                pyautogui.click(bx + bw // 2, by + bh // 2)
                break

        warp_roi = self.rh.load("warp_button")
        if warp_roi:
            cx = (warp_roi[0] + warp_roi[2]) // 2
            cy = (warp_roi[1] + warp_roi[3]) // 2
            pyautogui.click(cx, cy)
        self._sleep(2.0)

    def dock_or_undock(self, dock=True):
        key = "dock_button" if dock else "undock_button"
        box = self.rh.load(key)
        if box:
            x1, y1, x2, y2 = box
            pyautogui.click((x1 + x2) // 2, (y1 + y2) // 2)
            self._sleep(1.0)

    def refine_or_sell_ore(self):
        # Placeholder for refining/selling actions
        pass

    def log_statistics_and_events(self, message):
        print(f"[Mining] {message}")

    def human_like_idle(self):
        if random.random() < 0.3:
            self.randomize_camera_movement()
        self._sleep(random.uniform(0.5, 1.5))
