# version: 0.8.0
# path: src/bot_core.py

from __future__ import annotations

import sys
import time
import argparse
import re
import math
from typing import Any, Dict, List, Optional

from PySide6 import QtWidgets, QtGui
from .logger import get_logger

logger = get_logger(__name__)

from . import capture_utils
from .roi_capture import RegionHandler
from .ocr import OcrEngine
from .cv import CvEngine
from .state_machine import FSM, Event
from .mining_actions import MiningActions
from .env import EveEnv
from .agent import AIPilot
from .ui import Ui
from .llm_client import LMStudioClient


class EveBot:
    def __init__(
        self,
        model_path: Optional[str] = None,
        llm_enabled: Optional[bool] = None,
        llm_client: Optional[LMStudioClient] = None,
    ):
        # Initialize environment, agent, UI, and FSM
        self.env = EveEnv()
        self.fsm = FSM()
        self.agent = AIPilot(model_path=model_path, env=self.env, fsm=self.fsm)
        self.ui = Ui()
        self.running = False
        self.rh = RegionHandler()
        self.ocr = OcrEngine()
        self.cv = CvEngine()
        self.mining = MiningActions(
            ui=self.ui, region_handler=self.rh, ocr=self.ocr, cv=self.cv
        )
        self.gui_logger = None
        self.reward_label = None
        self.integrity_label = None

        self.mode = "auto"
        self.pending_action = None
        self.llm_config: Dict[str, Any] = dict(self.agent.config.get("llm", {}) or {})
        config_enabled = bool(self.llm_config.get("enabled", False))
        self.llm_planning_enabled = config_enabled if llm_enabled is None else bool(llm_enabled)
        self.llm_client: Optional[LMStudioClient] = llm_client or (
            self._build_llm_client() if self.llm_planning_enabled else None
        )
        self.last_llm_failure = 0.0
        self.last_plan: Optional[List[Dict[str, Any]]] = None
        self.llm_status_callback = None
        self._last_reward: Optional[float] = None

    def _build_llm_client(self) -> Optional[LMStudioClient]:
        if not self.llm_config:
            return None
        base_url = str(self.llm_config.get("endpoint", "") or "").strip()
        plan_path = str(self.llm_config.get("plan_path", "/v1/chat/completions") or "/v1/chat/completions")
        model = str(self.llm_config.get("model", "lmstudio") or "lmstudio")
        temperature = self.llm_config.get("temperature", 0.2)
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            temperature = 0.2
        timeout = self.llm_config.get("timeout", 10.0)
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = 10.0
        system_prompt = self.llm_config.get("system_prompt")
        try:
            return LMStudioClient(
                base_url=base_url or "http://localhost:1234",
                plan_path=plan_path,
                model=model,
                temperature=temperature,
                timeout=timeout,
                system_prompt=system_prompt,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to initialise LM Studio client: %s", exc)
            return None

    def log(self, message, level="info"):
        getattr(logger, level, logger.info)(message)
        timestamped = f"[{time.strftime('%H:%M:%S')}] {message}"
        if self.gui_logger:
            self.gui_logger.append(timestamped)

    def set_mode(self, mode: str) -> None:
        if mode not in {"auto", "manual", "assist"}:
            logger.warning("Unknown mode '%s'", mode)
            return
        self.mode = mode
        self.log(f"Mode switched to {mode}")

    def set_llm_planning(self, enabled: bool) -> None:
        if bool(enabled) == self.llm_planning_enabled:
            return
        self.llm_planning_enabled = bool(enabled)
        if self.llm_planning_enabled and self.llm_client is None:
            self.llm_client = self._build_llm_client()
        state = "enabled" if self.llm_planning_enabled else "disabled"
        self.log(f"LLM planning {state}.")
        if self.llm_status_callback:
            try:
                self.llm_status_callback(self.llm_planning_enabled)
            except Exception as exc:  # pragma: no cover - UI guard
                logger.debug("LLM status callback failed: %s", exc)

    def _format_detection(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        bbox = detection.get("bbox")
        if isinstance(bbox, (list, tuple)):
            try:
                bbox_list = [float(v) for v in bbox]
            except (TypeError, ValueError):
                bbox_list = list(bbox)
        else:
            bbox_list = None
        conf = detection.get("confidence")
        try:
            confidence = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            confidence = None
        return {
            "roi": detection.get("roi"),
            "label": detection.get("name"),
            "confidence": confidence,
            "bbox": bbox_list,
            "class_id": detection.get("class_id"),
        }

    def _clip_numeric(
        self, value: Optional[Any], minimum: Optional[float] = -1e6, maximum: Optional[float] = 1e6
    ) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        if minimum is not None:
            numeric = max(numeric, float(minimum))
        if maximum is not None:
            numeric = min(numeric, float(maximum))
        if isinstance(value, int):
            return int(round(numeric))
        return numeric

    def _scan_hostiles(self, screen) -> Optional[bool]:
        if screen is None:
            return None
        try:
            return bool(self.mining.detect_hostiles(screen))
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Hostile scan failed: %s", exc)
            return None

    def _scan_cargo_status(self, screen) -> Optional[int]:
        if screen is None:
            return None
        cargo_box = self.rh.load("mining_cargo_hold_capacity")
        if not cargo_box:
            return None
        x1, y1, x2, y2 = cargo_box
        crop = screen[y1:y2, x1:x2]
        try:
            text = self.ocr.extract_text(crop)
        except Exception as exc:  # pragma: no cover - OCR guard
            logger.debug("Cargo OCR failed: %s", exc)
            return None
        match = re.search(r"(\d+)", text)
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    def _scan_module_activity(self, screen) -> Optional[Dict[str, Any]]:
        if screen is None:
            return None
        slots = ["module_slot1", "module_slot2", "module_slot3"]
        slot_states: Dict[str, Dict[str, Any]] = {}
        any_active = False
        for slot in slots:
            box = self.rh.load(slot)
            if not box:
                continue
            x1, y1, x2, y2 = box
            try:
                active = bool(self.cv.is_module_active(screen[y1:y2, x1:x2]))
            except Exception as exc:  # pragma: no cover - vision guard
                logger.debug("Module scan failed for %s: %s", slot, exc)
                active = None
            slot_states[slot] = {"active": active, "box": box}
            if active:
                any_active = True
        if not slot_states:
            return None
        return {"any_active": any_active, "slots": slot_states}

    def _scan_target_status(self, screen) -> Optional[bool]:
        if screen is None:
            return None
        box = self.rh.load("is_target_locked")
        if not box:
            return None
        x1, y1, x2, y2 = box
        try:
            return bool(self.cv.detect_target_lock(screen[y1:y2, x1:x2]))
        except Exception as exc:  # pragma: no cover - vision guard
            logger.debug("Target lock scan failed: %s", exc)
            return None

    def _gather_perception(self, screen) -> Dict[str, Any]:
        observation = self.env.get_observation()
        ocr_excerpt = ""
        if screen is not None:
            try:
                ocr_excerpt = self.ocr.extract_text(screen)
            except Exception as exc:  # pragma: no cover - OCR fallback
                logger.debug("OCR extraction failed: %s", exc)
        try:
            templates = self.env._load_templates()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Template loading failed: %s", exc)
            templates = None
        detections: List[Dict[str, Any]] = []
        if templates and screen is not None:
            try:
                detections = self.cv.detect_elements(
                    screen,
                    templates,
                    threshold=self.env._detector_threshold(),
                )
            except Exception as exc:  # pragma: no cover - detector guard
                logger.debug("Detection failed: %s", exc)
                detections = []
        formatted = [self._format_detection(det) for det in detections][:10]
        cargo_pct = self._scan_cargo_status(screen) if screen is not None else None
        module_activity = self._scan_module_activity(screen) if screen is not None else None
        target_locked = self._scan_target_status(screen) if screen is not None else None
        hostiles_present = self._scan_hostiles(screen) if screen is not None else None
        status_modules = None
        if module_activity:
            status_modules = {
                "any_active": bool(module_activity.get("any_active")),
                "slots": {
                    slot: info.get("active")
                    for slot, info in module_activity.get("slots", {}).items()
                },
            }
        perception_status = {
            "cargo": {
                "percent": self._clip_numeric(cargo_pct, 0.0, 100.0)
                if cargo_pct is not None
                else None
            },
            "modules": status_modules,
            "hostiles": {
                "present": None if hostiles_present is None else bool(hostiles_present)
            },
            "target": {
                "locked": None if target_locked is None else bool(target_locked)
            },
            "reward": self._clip_numeric(self._last_reward) if self._last_reward is not None else None,
        }
        return {
            "timestamp": time.time(),
            "mode": self.mode,
            "state": self.fsm.state.name if self.fsm else None,
            "observation": observation,
            "ocr_excerpt": ocr_excerpt[:500],
            "detections": formatted,
            "status": perception_status,
        }

    def _execute_plan(self, plan: List[Dict[str, Any]]) -> None:
        for raw_action in plan:
            action = raw_action
            if not isinstance(action, dict):
                action = {"type": action}
            action_type = str(action.get("type", "")).lower()
            if action_type in {"comment", "message", "note"}:
                text = action.get("text") or action.get("message") or action.get("content")
                if text:
                    self.log(f"💬 LLM: {text}")
                continue
            if action_type == "noop":
                continue
            self.ui.execute(action)
            post_delay = action.get("sleep_after") or action.get("delay_after")
            if post_delay is not None:
                try:
                    delay = float(post_delay)
                except (TypeError, ValueError):
                    delay = 0.0
                if delay > 0:
                    time.sleep(delay)

    def _summarise_plan(self, plan: List[Dict[str, Any]]) -> str:
        labels: List[str] = []
        for action in plan:
            if isinstance(action, dict):
                labels.append(str(action.get("type", "noop")))
            else:
                labels.append(str(action))
            if len(labels) >= 5:
                break
        if len(plan) > 5:
            labels.append("…")
        return ", ".join(labels)

    def start(self):
        self.running = True
        self.fsm.on_event(Event.START_MINING)
        llm_state = "enabled" if self.llm_planning_enabled and self.llm_client else "disabled"
        self.log(f"▶️ Bot started. State: {self.fsm.state.name} (LLM {llm_state})")
        self._main_loop()

    def stop(self):
        self.running = False
        self.log("⛔ Bot stopped.")

    def manual_action(self, action_idx: int):
        """Execute a manual action index via the environment."""
        cmd = self.env._action_to_command(action_idx)
        self.ui.execute(cmd)
        reward = self.env._compute_reward()
        self.log(f"Manual action {action_idx} → reward {reward:.2f}")
        if self.reward_label:
            self.reward_label.setText(f"Reward: {reward:.2f}")

    def confirm_suggestion(self):
        if self.pending_action is not None:
            self.manual_action(self.pending_action)
            self.pending_action = None

    def _main_loop(self):
        while self.running:
            if self.mode == "manual":
                time.sleep(0.1)
                continue
            if self.mode == "assist":
                if self.pending_action is None:
                    obs = self.env.get_observation()["obs"]
                    self.pending_action = self.agent.bc_predict(obs)
                    self.log(f"Suggested action {self.pending_action}")
                time.sleep(0.1)
                continue

            screen = capture_utils.capture_screen(select_region=False)
            executed_plan = False
            if self.llm_planning_enabled:
                if self.llm_client is None:
                    self.llm_client = self._build_llm_client()
                if self.llm_client:
                    perception = self._gather_perception(screen)
                    plan = self.llm_client.plan_actions(perception)
                    if plan:
                        self.last_plan = plan
                        summary = self._summarise_plan(plan)
                        self.log(f"🤖 LLM plan: {summary}")
                        self._execute_plan(plan)
                        executed_plan = True
                    else:
                        self.last_plan = None
                        now = time.time()
                        if now - self.last_llm_failure > 10.0:
                            self.log("LLM planning unavailable, using heuristic fallback.", "warning")
                            self.last_llm_failure = now
                else:
                    now = time.time()
                    if now - self.last_llm_failure > 10.0:
                        self.log("LM Studio client not initialised; continuing without LLM.", "warning")
                        self.last_llm_failure = now

            if not executed_plan and self.fsm.state.name == "MINING":
                self._do_mining_routine(screen)

            reward = self.env._compute_reward()
            self._last_reward = self._clip_numeric(reward)
            if self.reward_label:
                self.reward_label.setText(f"Reward: {reward:.2f}")
            time.sleep(0.2)

    def _do_mining_routine(self, screen):
        self.log("⛏ Mining routine tick")

        # 0. HOSTILE CHECK
        hostiles_present = self._scan_hostiles(screen)
        if hostiles_present:
            self.log("⚠️ Hostiles detected")

        # 1. CARGO HOLD CHECK
        cargo_pct = self._scan_cargo_status(screen)
        if cargo_pct is not None:
            cargo_pct = int(self._clip_numeric(cargo_pct, 0.0, 100.0))
            self.log(f"📦 Cargo: {cargo_pct}%")
            if cargo_pct >= 90:
                self.log("🚀 Cargo full, docking...")
                self.fsm.on_event(Event.DOCK)
                self.mining.warp_to_station()
                self.mining.dock_or_undock(dock=True)
                return

        # 2. LASER MODULES
        module_activity = self._scan_module_activity(screen)
        slots = ["module_slot1", "module_slot2", "module_slot3"]
        active = bool(module_activity and module_activity.get("any_active"))

        if not active:
            self.log("🔄 Mining lasers inactive → activating")
            for slot in slots:
                box = None
                if module_activity and slot in module_activity.get("slots", {}):
                    box = module_activity["slots"][slot].get("box")
                if box is None:
                    box = self.rh.load(slot)
                if box:
                    x1, y1, x2, y2 = box
                    self.ui.click((x1 + x2) // 2, (y1 + y2) // 2)
                    time.sleep(0.2)
            time.sleep(1.0)
            return

        # 3. TARGET LOCK
        locked = self._scan_target_status(screen)

        if not locked:
            self.log("🔎 No target locked — acquiring new asteroid")
            self.mining.warp_to_asteroid_belt()

            # Sort overview by distance
            box = self.rh.load("overview_distance_header")
            if box:
                x1, y1, x2, y2 = box
                self.ui.click((x1 + x2) // 2, (y1 + y2) // 2)
                self.log("↕️ Sorting overview by distance")
                time.sleep(0.5)

            # Select asteroid using OCR fallback to first row
            if self.mining.target_asteroid_via_ocr():
                self.log("🪨 Selected asteroid via OCR")
            else:
                box = self.rh.load("overview_panel")
                if box:
                    x1, y1, x2, y2 = box
                    self.ui.click(x1 + 40, y1 + 15, jitter=2)
                    self.log("🪨 Selected nearest asteroid")
                    time.sleep(0.5)

            self.mining.approach_asteroid()
            self.mining.human_like_idle()


class BotGui(QtWidgets.QWidget):
    def __init__(self, llm_enabled: Optional[bool] = None):
        super().__init__()
        if hasattr(self, "setWindowTitle"):
            self.setWindowTitle("EVE Bot Controller")
        layout = QtWidgets.QVBoxLayout(self)
        add = getattr(layout, "addWidget", None)

        self.log_area = QtWidgets.QTextEdit()
        if hasattr(self.log_area, "setReadOnly"):
            self.log_area.setReadOnly(True)
        if add:
            add(self.log_area)

        self.reward_label = QtWidgets.QLabel("Reward: 0.0")
        self.integrity_label = QtWidgets.QLabel("Integrity: 100")
        if add:
            add(self.reward_label)
            add(self.integrity_label)

        self.start_btn = QtWidgets.QPushButton("Start Bot")
        self.stop_btn = QtWidgets.QPushButton("Stop Bot")
        if add:
            add(self.start_btn)
            add(self.stop_btn)

        self.override_input = QtWidgets.QLineEdit()
        if hasattr(self.override_input, "setPlaceholderText"):
            self.override_input.setPlaceholderText("Action index")
        self.override_btn = QtWidgets.QPushButton("Send Manual Action")
        if add:
            add(self.override_input)
            add(self.override_btn)
        self.mode_label = QtWidgets.QLabel("Mode: Auto")
        if add:
            add(self.mode_label)
        if hasattr(QtWidgets, "QShortcut"):
            QtWidgets.QShortcut(
                QtGui.QKeySequence("F9"),
                self,
                activated=lambda: self._switch_mode("auto"),
            )
            QtWidgets.QShortcut(
                QtGui.QKeySequence("F10"),
                self,
                activated=lambda: self._switch_mode("manual"),
            )
            QtWidgets.QShortcut(
                QtGui.QKeySequence("F11"),
                self,
                activated=lambda: self._switch_mode("assist"),
            )
            QtWidgets.QShortcut(
                QtGui.QKeySequence("F12"), self, activated=self._confirm_pending
            )

        self.bot = EveBot(model_path=None, llm_enabled=llm_enabled)
        self.bot.gui_logger = self.log_area
        self.bot.reward_label = self.reward_label

        if hasattr(self.start_btn, "clicked") and hasattr(
            self.start_btn.clicked, "connect"
        ):
            self.start_btn.clicked.connect(self.bot.start)
        if hasattr(self.stop_btn, "clicked") and hasattr(
            self.stop_btn.clicked, "connect"
        ):
            self.stop_btn.clicked.connect(self.bot.stop)
        if hasattr(self.override_btn, "clicked") and hasattr(
            self.override_btn.clicked, "connect"
        ):
            self.override_btn.clicked.connect(self._send_manual)

        self.llm_label = QtWidgets.QLabel()
        if add:
            add(self.llm_label)
        self._update_llm_label(self.bot.llm_planning_enabled)
        self.bot.llm_status_callback = self._update_llm_label

    def _switch_mode(self, mode):
        self.bot.set_mode(mode)
        if hasattr(self.mode_label, "setText"):
            self.mode_label.setText(f"Mode: {mode.capitalize()}")

    def _update_llm_label(self, enabled: bool) -> None:
        status = "On" if enabled else "Off"
        if hasattr(self.llm_label, "setText"):
            self.llm_label.setText(f"LLM planning: {status}")

    def _confirm_pending(self):
        self.bot.confirm_suggestion()

    def _send_manual(self):
        text = self.override_input.text()
        if text.isdigit():
            self.bot.manual_action(int(text))
        else:
            self.log_area.append("Invalid action index")


def main():
    parser = argparse.ArgumentParser(description="Run EveBot or BC inference")
    parser.add_argument(
        "--mode",
        choices=["gui", "bc_inference"],
        default="gui",
        help="Execution mode",
    )
    parser.add_argument(
        "--bc_model",
        type=str,
        default=None,
        help="Path to a trained BC model for inference",
    )
    parser.add_argument(
        "--llm-planning",
        dest="llm_planning",
        action="store_true",
        help="Enable LM Studio planning overrides",
    )
    parser.add_argument(
        "--no-llm-planning",
        dest="llm_planning",
        action="store_false",
        help="Disable LM Studio planning overrides",
    )
    parser.set_defaults(llm_planning=None)
    args = parser.parse_args()

    if args.mode == "gui":
        app = QtWidgets.QApplication(sys.argv)
        window = BotGui(llm_enabled=args.llm_planning)
        window.show()
        sys.exit(app.exec())

    if args.mode == "bc_inference":
        pilot = AIPilot(bc_model_path=args.bc_model)
        env = pilot.env
        obs = env.reset()
        done = False
        while not done:
            action = pilot.bc_predict(obs)
            obs, reward, done, _ = env.step(action)
            logger.info(f"Action: {action}, Reward: {reward:.2f}")


if __name__ == "__main__":
    main()
