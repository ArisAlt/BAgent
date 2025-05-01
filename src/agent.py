# version: 0.2.0
# path: src/agent.py

import os
import yaml
import pytesseract
import pyautogui

from capture_utils import capture_screen
from roi_capture import RegionHandler
from state_machine import State, Event

class AIPilot:
    """
    The agent driving EveBot. Loads constants from config/agent_config.yaml,
    uses RegionHandler for ROI lookups, and implements state-specific decision logic.
    """

    def __init__(self, config_path=None, region_handler=None):
        # Load configuration
        cfg_path = config_path or os.path.join(
            os.path.dirname(__file__),
            os.pardir, "config", "agent_config.yaml"
        )
        with open(cfg_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # ROI handler (for all click/hotkey targets)
        self.rh = region_handler or RegionHandler()

        # Internal flag to mark that station bookmark has been clicked
        self._station_selected = False

    def decide(self, obs, state):
        """
        Route to the appropriate state-specific method.
        """
        if state == State.MINING:
            return self.decide_mining(obs)
        elif state == State.DOCKING:
            return self.decide_docking(obs)
        else:
            return self.decide_idle(obs)

    def decide_mining(self, obs):
        """
        Mining logic:
        1) Target an asteroid if none is selected.
        2) Activate mining lasers if not already on.
        3) When cargo >= threshold, transition to docking via shortcut.
        """
        mining_cfg = self.config['mining']
        threshold = mining_cfg['cargo_threshold_pct']
        asteroid_roi = mining_cfg['asteroid_roi']
        module_keys = mining_cfg['mining_modules']['hotkeys']

        # 1) Target asteroid
        if not obs.get('asteroid_targeted', False):
            return {'type':'click', 'roi': asteroid_roi}

        # 2) Activate lasers
        if not obs.get('mining_laser_active', False):
            return {'type':'hotkey', 'keys': module_keys}

        # 3) Check cargo
        if obs.get('cargo_fill_pct', 0) >= threshold:
            # fire FSM event and dock via shortcut
            self.fsm.on_event(Event.DOCK)
            return self.dock_via_shortcut(obs)

        # Otherwise, do nothing and let the mining cycle continue
        return {'type':'noop'}

    def decide_docking(self, obs):
        """
        Docking logic via the in-game 'D' shortcut:
        Always use the shortcut routine when in DOCKING state.
        """
        return self.dock_via_shortcut(obs)

    def dock_via_shortcut(self, obs):
        """
        1) Open the Locations window
        2) OCR-scan for the station bookmark label
        3) Click the bookmark
        4) Press 'D' to dock
        """
        loc_cfg = self.config['locations']

        # 1) Open Locations panel
        roi = loc_cfg['window_roi']
        x1, y1, x2, y2 = self.rh.load(roi)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        self.click_at(cx, cy)

        # 2) OCR the panel crop
        full = capture_screen(select_region=False)
        panel = full[y1:y2, x1:x2]
        data = pytesseract.image_to_data(panel, output_type=pytesseract.Output.DICT)

        # 3) Find & click the station bookmark
        target = loc_cfg['station_bookmark']
        for i, txt in enumerate(data['text']):
            if txt.strip() == target:
                bx = x1 + data['left'][i]
                by = y1 + data['top'][i]
                bw = data['width'][i]
                bh = data['height'][i]
                self.click_at(bx + bw//2, by + bh//2)
                self._station_selected = True
                break

        # 4) Press 'D' to dock via shortcut
        return {'type':'hotkey', 'keys': ['d']}

    def decide_idle(self, obs):
        """
        Idle state: no specific action. Could be used for undocking or system checks.
        """
        return {'type':'noop'}

    # ---- Utility methods ----

    def click_at(self, x, y):
        """
        Move mouse to (x,y) and click.
        """
        pyautogui.moveTo(x, y)
        pyautogui.click()

    def press_hotkey(self, key):
        """
        Press a single key (or sequence).
        """
        pyautogui.press(key)

    # Optionally, you can add helpers to interpret and execute the returned action dict
