# version: 0.4.4
# path: src/env.py

import gym
from gym import spaces
import numpy as np
from ocr import OcrEngine
from cv import CvEngine
from ui import Ui
from roi_capture import RegionHandler

class EveEnv(gym.Env):
    def __init__(self, reward_config=None):
        super(EveEnv, self).__init__()
        self.ocr = OcrEngine()
        self.cv = CvEngine()
        self.ui = Ui()
        self.region_handler = RegionHandler()

        # Load all ROI names from YAML
        all_rois = self.region_handler.list_regions()
        # Partition ROIs by type
        self.click_rois = [r for r in all_rois if self.region_handler.get_type(r)=='click']
        self.text_rois = [r for r in all_rois if self.region_handler.get_type(r)=='text']
        self.detect_rois = [r for r in all_rois if self.region_handler.get_type(r)=='detect']

        # Define key actions
        self.key_actions = ['f1', 'f2', 'f3', 'f4', 'f5']
        # Build action list: clicks only for click_rois
        self.actions = [('click', roi) for roi in self.click_rois] + \
                       [('keypress', key) for key in self.key_actions] + \
                       [('sleep', None)]
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space: feature vector
        obs_dim = 64 + len(self.text_rois)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

        # Reward config
        defaults = {'mine_active': 1.0, 'cargo_increase': 5.0,
                    'hostile_penalty': -10.0, 'cargo_full': 20.0}
        self.reward_config = reward_config or defaults

        # Init cargo tracking
        vol, cap = self._read_cargo_capacity()
        self.prev_volume, self.cargo_capacity = vol, cap

    def reset(self):
        vol, cap = self._read_cargo_capacity()
        self.prev_volume, self.cargo_capacity = vol, cap
        return self._get_obs()

    def step(self, action):
        cmd = self._action_to_command(action)
        self.ui.execute(cmd)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        return obs, reward, done, {}

    def _get_obs(self):
        img = self.ui.capture()
        # Basic features: OCR text length, number of detect elements
        text_full = self.ocr.extract_text(img)
        elems = self.cv.detect_elements(img, templates=self._load_templates())
        base_feats = [len(text_full), len(elems)]
        # Text ROI readings
        text_vals = []
        for roi in self.text_rois:
            coords = self.region_handler.get_coords(roi)
            if coords:
                x1,y1,x2,y2 = coords
                sub = img[y1:y2, x1:x2]
                val = self.ocr.extract_text(sub)
                text_vals.append(len(val))
            else:
                text_vals.append(0)
        obs = np.array(base_feats + text_vals, dtype=np.float32)
        # Pad to obs_dim
        vec = np.zeros(self.observation_space.shape, dtype=np.float32)
        vec[:len(obs)] = obs
        return vec

    def _load_templates(self):
        # Only detect actions for detect_rois
        return {name: f"templates/{name}.png" for name in self.detect_rois}

    def _action_to_command(self, action):
        cmd_type, target = self.actions[action]
        if cmd_type == 'click':
            coords = self.region_handler.get_coords(target)
            if coords:
                x1,y1,x2,y2 = coords
                cx, cy = (x1+x2)//2, (y1+y2)//2
                return {'type':'click','x':cx,'y':cy}
        if cmd_type == 'keypress':
            return {'type':'keypress','key':target}
        return {'type':'sleep','duration':1.0}

    def _read_cargo_capacity(self):
        coords = self.region_handler.get_coords('cargo_hold')
        img = self.ui.capture()
        sub = img if coords is None else img[coords[1]:coords[3], coords[0]:coords[2]]
        text = self.ocr.extract_text(sub)
        nums = [int(s) for s in text.replace('/',' ').split() if s.isdigit()]
        if len(nums)>=2:
            return nums[0], nums[1]
        return (nums[0], nums[0]) if nums else (0,1)

    def _compute_reward(self):
        reward = 0.0
        img = self.ui.capture()
        # Detect mining
        detect = self.cv.detect_elements(img, templates=self._load_templates())
        if any(e['name']=='mining_laser_on' for e in detect):
            reward += self.reward_config['mine_active']
        # Cargo
        vol,_ = self._read_cargo_capacity()
        if vol>self.prev_volume:
            reward += self.reward_config['cargo_increase']
        if vol>=self.cargo_capacity:
            reward += self.reward_config['cargo_full']
        self.prev_volume=vol
        # Hostile
        if any(e['name']=='hostile_alert' for e in detect):
            reward += self.reward_config['hostile_penalty']
        return reward

    def _check_done(self):
        img = self.ui.capture()
        text = self.ocr.extract_text(img).lower()
        return ('destroyed' in text) or ('docked' in text)

    def get_observation(self):
        """Return a dictionary snapshot of the current observation state."""
        vec = self._get_obs()
        return {
            'obs': vec.tolist(),
            'prev_volume': getattr(self, 'prev_volume', None),
            'cargo_capacity': getattr(self, 'cargo_capacity', None)
        }
