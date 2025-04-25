# version: 0.3.5
# path: src/ui.py

import pyautogui
import numpy as np
import time
import threading
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

    def click(self, x, y, duration=0.1):
        with self.lock:
            pyautogui.moveTo(x, y, duration=0.1)
            pyautogui.click()
            time.sleep(duration)

    def key_press(self, key, duration=0.1):
        with self.lock:
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)

    def execute(self, command):
        cmd_type = command.get('type')
        if cmd_type == 'click':
            self.click(command['x'], command['y'])
        elif cmd_type == 'keypress':
            self.key_press(command['key'])
        elif cmd_type == 'sleep':
            time.sleep(command.get('duration', 1.0))
        elif cmd_type == 'switch_region':
            self.load_capture_region(command['region_name'])

    def load_capture_region(self, region_name, yaml_path='regions.yaml'):
        region = self.region_handler.load(region_name, yaml_path)
        if region:
            self.capture_region = region
            self.region_handler.validate(self.capture_region)

    def load_preset_region(self, preset_name):
        presets = ['overview_panel', 'mining_lasers', 'cargo_hold', 'system_status', 'hostile_warning', 'chat_window']
        if preset_name in presets:
            self.load_capture_region(preset_name)
        else:
            print(f"Preset '{preset_name}' not recognized. Available presets: {presets}")
