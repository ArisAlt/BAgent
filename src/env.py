# version: 0.5.0
# path: src/env.py

try:
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - allow import without gym
    import types

    gym = types.SimpleNamespace(Env=object)

    class DummyDiscrete:
        def __init__(self, n):
            self.n = n

        def sample(self):
            return 0

    class DummyBox:
        def __init__(self, low, high, shape, dtype=None):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

        def sample(self):
            import numpy as np

            return np.zeros(self.shape, dtype=self.dtype or np.float32)

    spaces = types.SimpleNamespace(Discrete=DummyDiscrete, Box=DummyBox)
import numpy as np
from typing import Dict, Iterable, Optional, Set, Tuple

try:
    from .ocr import OcrEngine
except Exception:  # pragma: no cover - allow tests without PaddleOCR

    class OcrEngine:
        def extract_text(self, img):
            return ""


try:
    from .cv import CvEngine
except Exception:  # pragma: no cover - allow tests without OpenCV

    class CvEngine:
        def detect_elements(self, *a, **kw):
            return []


from .ui import Ui
from .roi_capture import RegionHandler
from .config import get_window_title
from .detector import load_detector_settings, map_roi_labels


class EveEnv(gym.Env):
    def __init__(self, reward_config=None, window_title=None):
        if window_title is None:
            window_title = get_window_title()
        super(EveEnv, self).__init__()
        self.ocr = OcrEngine()
        try:
            self.cv = CvEngine()
        except Exception:  # pragma: no cover - allow tests without cv2

            class CvEngine:
                def detect_elements(self, *a, **kw):
                    return []

            self.cv = CvEngine()
        self.ui = Ui(window_title=window_title)
        self.region_handler = RegionHandler()
        self.detector_cfg = load_detector_settings()

        # Load all ROI names from YAML
        all_rois = self.region_handler.list_regions()
        # Partition ROIs by type
        self.click_rois = [
            r for r in all_rois if self.region_handler.get_type(r) == "click"
        ]
        self.text_rois = [
            r for r in all_rois if self.region_handler.get_type(r) == "text"
        ]
        self.detect_rois = [
            r for r in all_rois if self.region_handler.get_type(r) == "detect"
        ]

        # Define key actions
        self.key_actions = ["f1", "f2", "f3", "f4", "f5"]
        # Build action list: clicks only for click_rois
        self.actions = (
            [("click", roi) for roi in self.click_rois]
            + [("keypress", key) for key in self.key_actions]
            + [("sleep", None)]
        )
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space: feature vector
        obs_dim = 64 + len(self.text_rois)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Reward config
        defaults = {
            "mine_active": 1.0,
            "cargo_increase": 5.0,
            "hostile_penalty": -10.0,
            "cargo_full": 20.0,
        }
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
        targets = self._load_templates()
        elems = self.cv.detect_elements(
            img,
            templates=targets,
            threshold=self._detector_threshold(),
        )
        base_feats = [len(text_full), len(elems)]
        # Text ROI readings
        text_vals = []
        for roi in self.text_rois:
            coords = self.region_handler.get_coords(roi)
            if coords:
                x1, y1, x2, y2 = coords
                sub = img[y1:y2, x1:x2]
                val = self.ocr.extract_text(sub)
                text_vals.append(len(val))
            else:
                text_vals.append(0)
        obs = np.array(base_feats + text_vals, dtype=np.float32)
        # Pad to obs_dim
        vec = np.zeros(self.observation_space.shape, dtype=np.float32)
        vec[: len(obs)] = obs
        return vec

    def _load_templates(self):
        entries = getattr(self.region_handler, "regions", {})
        label_map = map_roi_labels(self.detect_rois, entries, self.detector_cfg)
        override_map = (
            self.detector_cfg.get("roi_map", {})
            if isinstance(self.detector_cfg, dict)
            else {}
        )
        targets = {}
        for name in self.detect_rois:
            coords = self.region_handler.get_coords(name)
            override = override_map.get(name, {}) if isinstance(override_map, dict) else {}
            roi_override = override.get("roi") if isinstance(override, dict) else None
            roi_coords: Optional[Tuple[int, int, int, int]] = None
            source_coords = roi_override or coords
            if isinstance(source_coords, (list, tuple)) and len(source_coords) == 4:
                roi_coords = tuple(int(v) for v in source_coords)
            labels = label_map.get(name, {}).get("labels") if label_map else None
            if labels:
                labels = [str(v) for v in labels]
            target_entry: Dict[str, object] = {
                "roi": roi_coords,
                "labels": labels,
            }
            template_path = override.get("template") if isinstance(override, dict) else None
            if template_path:
                target_entry["template"] = template_path
            targets[name] = target_entry
        return targets

    def _detector_threshold(self) -> float:
        if isinstance(self.detector_cfg, dict):
            try:
                return float(self.detector_cfg.get("default_threshold", 0.25))
            except (TypeError, ValueError):
                pass
        return 0.25

    def _reward_labels(self, key: str, default: Iterable[str]) -> Set[str]:
        labels = {str(v) for v in default}
        if isinstance(self.detector_cfg, dict):
            reward_cfg = self.detector_cfg.get("reward_labels", {})
            if isinstance(reward_cfg, dict):
                cfg_labels = reward_cfg.get(key)
                if isinstance(cfg_labels, (list, tuple, set)):
                    labels = {str(v) for v in cfg_labels}
        return labels

    def _action_to_command(self, action):
        cmd_type, target = self.actions[action]
        if cmd_type == "click":
            coords = self.region_handler.get_coords(target)
            if coords:
                x1, y1, x2, y2 = coords
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                return {"type": "click", "x": cx, "y": cy}
        if cmd_type == "keypress":
            return {"type": "keypress", "key": target}
        return {"type": "sleep", "duration": 1.0}

    def _read_cargo_capacity(self):
        coords = self.region_handler.get_coords("cargo_hold")
        img = self.ui.capture()
        sub = (
            img if coords is None else img[coords[1] : coords[3], coords[0] : coords[2]]
        )
        text = self.ocr.extract_text(sub)
        nums = [int(s) for s in text.replace("/", " ").split() if s.isdigit()]
        if len(nums) >= 2:
            return nums[0], nums[1]
        return (nums[0], nums[0]) if nums else (0, 1)

    def _compute_reward(self):
        reward = 0.0
        img = self.ui.capture()
        # Detect mining
        detect = self.cv.detect_elements(
            img,
            templates=self._load_templates(),
            threshold=self._detector_threshold(),
        )
        labels = {str(d.get("name")) for d in detect if d.get("name")}
        if labels & self._reward_labels("mine_active", ["mining_laser_on"]):
            reward += self.reward_config["mine_active"]
        # Cargo
        vol, _ = self._read_cargo_capacity()
        if vol > self.prev_volume:
            reward += self.reward_config["cargo_increase"]
        if vol >= self.cargo_capacity:
            reward += self.reward_config["cargo_full"]
        self.prev_volume = vol
        # Hostile
        if labels & self._reward_labels("hostile_penalty", ["hostile_alert"]):
            reward += self.reward_config["hostile_penalty"]
        return reward

    def _check_done(self):
        img = self.ui.capture()
        text = self.ocr.extract_text(img).lower()
        return ("destroyed" in text) or ("docked" in text)

    def get_observation(self):
        """Return a dictionary snapshot of the current observation state."""
        vec = self._get_obs()
        return {
            "obs": vec.tolist(),
            "prev_volume": getattr(self, "prev_volume", None),
            "cargo_capacity": getattr(self, "cargo_capacity", None),
        }
