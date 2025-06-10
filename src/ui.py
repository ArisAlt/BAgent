# version: 0.3.7
# path: src/ui.py

import pyautogui
import numpy as np
import time
import threading
import random
from capture_utils import capture_screen
from roi_capture import RegionHandler

class Ui:
    def __init__(self, capture_region=None):
        self.capture_region = capture_region
        self.region_handler = RegionHandler()
        self.lock = threading.Lock()

    def capture(self):
        with self.lock:
            return capture_screen(self.capture_region)

    def click(self, x, y, duration=0.1, jitter=5):
        with self.lock:
            jitter_x = x + random.randint(-jitter, jitter)
            jitter_y = y + random.randint(-jitter, jitter)
            delay = duration + random.uniform(-0.05, 0.05)
            pyautogui.moveTo(jitter_x, jitter_y, duration=delay)
            pyautogui.click()
            time.sleep(delay)

    def key_press(self, key, duration=0.1):
        with self.lock:
            delay = duration + random.uniform(-0.05, 0.05)
            pyautogui.keyDown(key)
            time.sleep(delay)
            pyautogui.keyUp(key)

    def execute(self, command):
        cmd_type = command.get('type')
        if cmd_type == 'click':
            self.click(command['x'], command['y'])
        elif cmd_type == 'keypress':
            self.key_press(command['key'])
        elif cmd_type == 'sleep':
            delay = command.get('duration', 1.0) + random.uniform(-0.1, 0.1)
            time.sleep(delay)
        elif cmd_type == 'switch_region':
            self.load_capture_region(command['region_name'])

    def load_capture_region(self, region_name, yaml_path='regions.yaml'):
        """Load a named region from the YAML file and set it as capture region."""
        # Reinitialize handler when a custom YAML path is provided
        if yaml_path:
            self.region_handler = RegionHandler(yaml_path)
        region = self.region_handler.load(region_name)
        if region:
            self.capture_region = region

    def load_preset_region(self, preset_name):
        presets = ['overview_panel', 'mining_lasers', 'cargo_hold', 'system_status', 'hostile_warning', 'chat_window']
        if preset_name in presets:
            self.load_capture_region(preset_name)
        else:
            print(f"Preset '{preset_name}' not recognized. Available presets: {presets}")
