# version: 0.3.0
# path: src/agent.py

import os
import yaml
import pytesseract
import pyautogui
from stable_baselines3 import PPO

from capture_utils import capture_screen
from roi_capture import RegionHandler
from state_machine import State, Event
from env import EveEnv

class AIPilot:
    """
    The agent driving EveBot. Loads constants from config/agent_config.yaml,
    uses RegionHandler for ROI lookups, and implements state-specific decision logic.
    """

    def __init__(self, config_path=None, region_handler=None, model_path=None, env=None, fsm=None):
        # Load configuration
        cfg_path = config_path or os.path.join(
            os.path.dirname(__file__),
            os.pardir, "config", "agent_config.yaml"
        )
        with open(cfg_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # ROI handler (for all click/hotkey targets)
        self.rh = region_handler or RegionHandler()
        self.env = env or EveEnv()
        self.fsm = fsm
        self.model = None
        if model_path is None:
            model_path = self._find_latest_model()
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(model_path, env=self.env)

        # Internal flag to mark that station bookmark has been clicked
        self._station_selected = False

    def set_fsm(self, fsm):
        """Attach a finite state machine to the pilot after construction."""
        self.fsm = fsm

    def decide(self, obs=None, state=None):
        """Return a command based on either the RL model or rule logic."""
        if self.model:
            if obs is None:
                obs = self.env._get_obs()
            action, _ = self.model.predict(obs, deterministic=True)
            return self.env._action_to_command(int(action))

        if state == State.MINING:
            return self.decide_mining(obs)
        elif state == State.DOCKING:
            return self.decide_docking(obs)
        elif state == State.COMBAT:
            return self.decide_combat(obs)
        elif state == State.MISSION:
            return self.decide_mission(obs)
        elif state == State.EXPLORATION:
            return self.decide_exploration(obs)
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
        threshold = mining_cfg.get('cargo_threshold_pct', mining_cfg.get('cargo_threshold', 90))
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

    def decide_combat(self, obs):
        """Simple combat routine: activate combat modules and retreat if low health."""
        combat_cfg = self.config['combat']
        retreat = combat_cfg.get('retreat_threshold', 30)
        modules = combat_cfg.get('activate_modules', [])

        if obs.get('ship_health_pct', 100) <= retreat:
            if self.fsm:
                self.fsm.on_event(Event.DOCK)
            return self.dock_via_shortcut(obs)

        if not obs.get('combat_modules_active', False) and modules:
            return {'type': 'hotkey', 'keys': modules}

        return {'type': 'noop'}

    def decide_mission(self, obs):
        """Mission acceptance logic."""
        mission_cfg = self.config['mission']
        if not obs.get('mission_accepted', False):
            roi = mission_cfg.get('accept_mission_roi')
            if roi:
                return {'type': 'click', 'roi': roi}
        return {'type': 'noop'}

    def decide_exploration(self, obs):
        """Exploration scanning routine."""
        exp_cfg = self.config['exploration']
        if not obs.get('anomaly_found', False):
            roi = exp_cfg.get('scan_button_roi')
            if roi:
                return {'type': 'click', 'roi': roi}
        else:
            roi = exp_cfg.get('anomaly_window_roi')
            if roi:
                return {'type': 'click', 'roi': roi}
        return {'type': 'noop'}

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

    def _find_latest_model(self):
        """Return path to most recently modified PPO model in logs dir."""
        log_dir = os.path.join(os.path.dirname(__file__), os.pardir, "logs")
        if not os.path.isdir(log_dir):
            return None
        candidates = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if f.endswith(".zip")
        ]
        if not candidates:
            return None
        latest = max(candidates, key=os.path.getmtime)
        return latest

    # Optionally, you can add helpers to interpret and execute the returned action dict
