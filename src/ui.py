# version: 0.5.0
# path: src/ui.py

import pyautogui
import numpy as np
import time
import threading
import random
from typing import Any, Dict, Iterable, Optional, Tuple

from .capture_utils import capture_screen
from .roi_capture import RegionHandler
from .config import get_window_title

class Ui:
    def __init__(self, capture_region=None, window_title=None):
        if window_title is None:
            window_title = get_window_title()
        self.capture_region = capture_region
        self.window_title = window_title
        self.region_handler = RegionHandler()
        self.lock = threading.Lock()

    def capture(self):
        with self.lock:
            frame = capture_screen(select_region=False, window_title=self.window_title)
            if frame is None:
                return None
            if self.capture_region:
                x1, y1, x2, y2 = self.capture_region
                frame = frame[y1:y2, x1:x2]
            return frame

    def click(
        self,
        x: float,
        y: float,
        duration: float = 0.1,
        jitter: int = 5,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.05,
    ) -> None:
        with self.lock:
            jitter_x = x + random.randint(-jitter, jitter)
            jitter_y = y + random.randint(-jitter, jitter)
            delay = max(duration + random.uniform(-0.05, 0.05), 0.0)
            pyautogui.moveTo(jitter_x, jitter_y, duration=delay)
            pyautogui.click(
                x=jitter_x,
                y=jitter_y,
                clicks=max(int(clicks), 1),
                interval=max(interval, 0.0),
                button=button,
            )
        if delay:
            time.sleep(delay)

    def key_press(self, key, duration=0.1):
        with self.lock:
            delay = duration + random.uniform(-0.05, 0.05)
            pyautogui.keyDown(key)
            time.sleep(delay)
            pyautogui.keyUp(key)

    def hotkey(self, *keys: str, interval: float = 0.0) -> None:
        with self.lock:
            pyautogui.hotkey(*keys, interval=interval)

    def move(self, x: float, y: float, duration: float = 0.1) -> None:
        with self.lock:
            pyautogui.moveTo(x, y, duration=duration)

    def drag_to(
        self,
        x: float,
        y: float,
        duration: float = 0.2,
        button: str = "left",
    ) -> None:
        with self.lock:
            pyautogui.dragTo(x, y, duration=duration, button=button)

    def _center_from_roi(self, roi_name: str) -> Optional[Tuple[int, int]]:
        coords = self.region_handler.load(roi_name)
        if not coords:
            return None
        x1, y1, x2, y2 = coords
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _resolve_coords(self, command: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        if "x" in command and "y" in command:
            try:
                return float(command["x"]), float(command["y"])
            except (TypeError, ValueError):
                return None
        roi_name = command.get("roi")
        if roi_name:
            return self._center_from_roi(roi_name)
        return None

    def execute(self, command: Dict[str, Any]) -> None:
        if not command:
            return

        cmd_type = str(command.get('type', '')).lower()
        if cmd_type in {'click', 'left_click', 'right_click', 'double_click'}:
            coords = self._resolve_coords(command)
            if coords:
                button = command.get('button')
                if not button:
                    button = 'right' if cmd_type == 'right_click' else 'left'
                clicks = command.get('clicks') or (2 if cmd_type == 'double_click' else 1)
                self.click(
                    coords[0],
                    coords[1],
                    duration=command.get('duration', 0.1),
                    jitter=command.get('jitter', 5),
                    button=button,
                    clicks=clicks,
                    interval=command.get('interval', 0.05),
                )
        elif cmd_type in {'coordinate_click', 'click_coords'}:
            coords = self._resolve_coords(command)
            if coords:
                self.click(
                    coords[0],
                    coords[1],
                    duration=command.get('duration', 0.1),
                    jitter=command.get('jitter', 0),
                    button=command.get('button', 'left'),
                    clicks=command.get('clicks', 1),
                    interval=command.get('interval', 0.05),
                )
        elif cmd_type in {'move', 'move_to'}:
            coords = self._resolve_coords(command)
            if coords:
                self.move(coords[0], coords[1], duration=command.get('duration', 0.1))
        elif cmd_type == 'drag':
            coords = self._resolve_coords(command)
            if coords:
                start = command.get('start')
                if isinstance(start, (list, tuple)) and len(start) == 2:
                    self.move(float(start[0]), float(start[1]), duration=command.get('pre_move', 0.1))
                self.drag_to(
                    coords[0],
                    coords[1],
                    duration=command.get('duration', 0.2),
                    button=command.get('button', 'left'),
                )
        elif cmd_type in {'keypress', 'key_press', 'press'}:
            key = command.get('key')
            if key:
                self.key_press(key, duration=command.get('duration', 0.1))
        elif cmd_type in {'enter', 'return'}:
            self.key_press('enter', duration=command.get('duration', 0.1))
        elif cmd_type in {'key_down', 'keydown'}:
            key = command.get('key')
            if key:
                with self.lock:
                    pyautogui.keyDown(key)
        elif cmd_type in {'key_up', 'keyup'}:
            key = command.get('key')
            if key:
                with self.lock:
                    pyautogui.keyUp(key)
        elif cmd_type == 'hotkey':
            keys = command.get('keys')
            if isinstance(keys, Iterable):
                self.hotkey(*[str(k) for k in keys], interval=command.get('interval', 0.0))
        elif cmd_type in {'type', 'typewrite', 'text'}:
            text = command.get('text')
            if text is not None:
                with self.lock:
                    pyautogui.typewrite(str(text), interval=command.get('interval', 0.05))
        elif cmd_type in {'scroll', 'mouse_scroll'}:
            amount = command.get('amount')
            if amount is not None:
                with self.lock:
                    pyautogui.scroll(int(amount))
        elif cmd_type in {'sleep', 'wait', 'delay'}:
            duration = command.get('duration', command.get('seconds', 1.0))
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                duration = 0.0
            duration += random.uniform(-0.1, 0.1)
            if duration > 0:
                time.sleep(max(duration, 0.0))
        elif cmd_type == 'switch_region':
            region_name = command.get('region_name') or command.get('region')
            if region_name:
                self.load_capture_region(region_name)
        elif cmd_type == 'sequence':
            for sub in command.get('actions', []):
                self.execute(sub)
        elif cmd_type == 'noop':
            return

    def load_capture_region(self, region_name, yaml_path='regions.yaml'):
        if yaml_path != self.region_handler.yaml_path:
            self.region_handler = RegionHandler(yaml_path)
        region = self.region_handler.load(region_name)
        if region:
            self.capture_region = region
            self.region_handler.validate(self.capture_region, save_preview=True, region_name=region_name)

    def load_preset_region(self, preset_name):
        presets = ['overview_panel', 'mining_lasers', 'cargo_hold', 'system_status', 'hostile_warning', 'chat_window']
        if preset_name in presets:
            self.load_capture_region(preset_name)
        else:
            print(f"Preset '{preset_name}' not recognized. Available presets: {presets}")
