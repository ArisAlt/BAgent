# version: 0.6.0
# path: src/ui.py

import pyautogui
import numpy as np
import time
import threading
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .capture_utils import capture_screen
from .roi_capture import RegionHandler
from .config import get_window_title


COMMAND_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "click",
        "aliases": ["left_click", "right_click", "double_click", "coordinate_click", "click_coords"],
        "description": "Move the cursor (optionally with jitter) and press the specified mouse button one or more times.",
        "parameters": [
            {
                "name": "x",
                "type": "float",
                "description": "Screen X coordinate. Required if 'roi' is not provided.",
            },
            {
                "name": "y",
                "type": "float",
                "description": "Screen Y coordinate. Required if 'roi' is not provided.",
            },
            {
                "name": "roi",
                "type": "str",
                "description": "Named region to resolve to a centre point instead of coordinates.",
            },
            {
                "name": "duration",
                "type": "float",
                "description": "Seconds to move before clicking (default 0.1).",
            },
            {
                "name": "jitter",
                "type": "int",
                "description": "Random pixel jitter applied to the destination (default 5).",
            },
            {
                "name": "button",
                "type": "str",
                "description": "Mouse button to use ('left' or 'right'). Defaults by command type.",
            },
            {
                "name": "clicks",
                "type": "int",
                "description": "Number of click repetitions (double_click defaults to 2).",
            },
            {
                "name": "interval",
                "type": "float",
                "description": "Delay between multi-clicks (default 0.05).",
            },
        ],
    },
    {
        "type": "move",
        "aliases": ["move_to"],
        "description": "Move the mouse cursor without clicking.",
        "parameters": [
            {
                "name": "x",
                "type": "float",
                "description": "Screen X coordinate. Required if 'roi' is not provided.",
            },
            {
                "name": "y",
                "type": "float",
                "description": "Screen Y coordinate. Required if 'roi' is not provided.",
            },
            {
                "name": "roi",
                "type": "str",
                "description": "Named region to resolve to a centre point instead of coordinates.",
            },
            {
                "name": "duration",
                "type": "float",
                "description": "Seconds to complete the move (default 0.1).",
            },
        ],
    },
    {
        "type": "drag",
        "description": "Click-and-drag to a destination. Optionally move to a start point first.",
        "parameters": [
            {
                "name": "x",
                "type": "float",
                "description": "Destination X coordinate. Required if 'roi' is not provided.",
            },
            {
                "name": "y",
                "type": "float",
                "description": "Destination Y coordinate. Required if 'roi' is not provided.",
            },
            {
                "name": "roi",
                "type": "str",
                "description": "Destination ROI name to centre on.",
            },
            {
                "name": "start",
                "type": "[float, float]",
                "description": "Optional [x, y] pair for the drag start position before pressing down.",
            },
            {
                "name": "pre_move",
                "type": "float",
                "description": "Seconds used when moving to the start position (default 0.1).",
            },
            {
                "name": "duration",
                "type": "float",
                "description": "Seconds to complete the drag (default 0.2).",
            },
            {
                "name": "button",
                "type": "str",
                "description": "Mouse button to hold during the drag (default 'left').",
            },
        ],
    },
    {
        "type": "keypress",
        "aliases": ["key_press", "press"],
        "description": "Press and release a single key.",
        "parameters": [
            {
                "name": "key",
                "type": "str",
                "description": "Key name (required for generic keypress commands).",
            },
            {
                "name": "duration",
                "type": "float",
                "description": "Seconds to hold the key before release (default 0.1).",
            },
        ],
    },
    {
        "type": "enter",
        "aliases": ["return"],
        "description": "Press the Enter/Return key without requiring a 'key' parameter.",
        "parameters": [
            {
                "name": "duration",
                "type": "float",
                "description": "Seconds to hold the key before release (default 0.1).",
            }
        ],
    },
    {
        "type": "key_down",
        "aliases": ["keydown"],
        "description": "Hold a key down until a matching key_up is issued.",
        "parameters": [
            {"name": "key", "type": "str", "description": "Key name to hold."}
        ],
    },
    {
        "type": "key_up",
        "aliases": ["keyup"],
        "description": "Release a key previously held with key_down.",
        "parameters": [
            {"name": "key", "type": "str", "description": "Key name to release."}
        ],
    },
    {
        "type": "hotkey",
        "description": "Press a combination of keys in sequence (e.g., ctrl+s).",
        "parameters": [
            {
                "name": "keys",
                "type": "List[str]",
                "description": "Ordered keys to press (required).",
            },
            {
                "name": "interval",
                "type": "float",
                "description": "Delay between key presses (default 0).",
            },
        ],
    },
    {
        "type": "type",
        "aliases": ["typewrite", "text"],
        "description": "Type literal text.",
        "parameters": [
            {
                "name": "text",
                "type": "str",
                "description": "Text to type (required).",
            },
            {
                "name": "interval",
                "type": "float",
                "description": "Delay between characters (default 0.05).",
            },
        ],
    },
    {
        "type": "scroll",
        "aliases": ["mouse_scroll"],
        "description": "Scroll the mouse wheel.",
        "parameters": [
            {
                "name": "amount",
                "type": "int",
                "description": "Positive scrolls up, negative scrolls down (required).",
            }
        ],
    },
    {
        "type": "sleep",
        "aliases": ["wait", "delay"],
        "description": "Pause for a duration of time (random jitter applied).",
        "parameters": [
            {
                "name": "duration",
                "type": "float",
                "description": "Seconds to wait (preferred field).",
            },
            {
                "name": "seconds",
                "type": "float",
                "description": "Alternative field for wait duration.",
            },
        ],
    },
    {
        "type": "switch_region",
        "description": "Load a saved capture region to crop subsequent screenshots.",
        "parameters": [
            {
                "name": "region_name",
                "type": "str",
                "description": "Named capture region to load. 'region' is accepted as an alias field.",
            }
        ],
    },
    {
        "type": "sequence",
        "description": "Execute a nested list of sub-actions in order.",
        "parameters": [
            {
                "name": "actions",
                "type": "List[Dict]",
                "description": "Inline list of action objects to execute sequentially.",
            }
        ],
    },
    {
        "type": "noop",
        "description": "Do nothing (used as a placeholder).",
        "parameters": [],
    },
]


def summarise_command_schema(schema: Optional[List[Dict[str, Any]]] = None) -> str:
    """Return a human-readable summary of the supported command schema."""

    schema = schema or COMMAND_SCHEMA
    lines: List[str] = []
    for entry in schema:
        type_name = entry.get("type", "")
        if not type_name:
            continue
        aliases = entry.get("aliases") or []
        alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
        description = entry.get("description") or ""
        lines.append(f"- {type_name}{alias_text}: {description}")
        params = entry.get("parameters") or []
        if params:
            param_bits = []
            for param in params:
                name = param.get("name")
                if not name:
                    continue
                type_hint = param.get("type")
                desc = param.get("description")
                segment = name
                if type_hint:
                    segment += f" ({type_hint})"
                if desc:
                    segment += f" – {desc}"
                param_bits.append(segment)
            if param_bits:
                lines.append("  params: " + "; ".join(param_bits))
    return "\n".join(lines)

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
