# version: 0.5.3
# path: src/bot_core.py

import sys
import time
import argparse
import re
import pyautogui
import cv2
from PySide6 import QtWidgets, QtCore, QtGui
from logger import get_logger

logger = get_logger(__name__)

from roi_capture import RegionHandler
from ocr import OcrEngine
from cv import CvEngine
from state_machine import FSM, Event
from mining_actions import MiningActions
from env import EveEnv
from agent import AIPilot
from ui import Ui

class EveBot:
    def __init__(self, model_path=None):
        # Initialize environment, agent, UI, and FSM
        self.env = EveEnv()
        self.fsm = FSM()
        self.agent = AIPilot(model_path=model_path, env=self.env, fsm=self.fsm)
        self.ui = Ui()
        self.running = False
        self.rh = RegionHandler()
        self.ocr = OcrEngine()
        self.cv = CvEngine()
        self.mining = MiningActions(ui=self.ui,
                                    region_handler=self.rh,
                                    ocr=self.ocr,
                                    cv=self.cv)
        self.gui_logger = None
        self.reward_label = None
        self.integrity_label = None

    def log(self, message, level="info"):
        getattr(logger, level, logger.info)(message)
        timestamped = f"[{time.strftime('%H:%M:%S')}] {message}"
        if self.gui_logger:
            self.gui_logger.append(timestamped)

    def start(self):
        self.running = True
        self.fsm.on_event(Event.START_MINING)
        self.log(f"▶️ Bot started. State: {self.fsm.state.name}")
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

    def _main_loop(self):
        import capture_utils
        while self.running:
            screen = capture_utils.capture_screen(select_region=False)
            if self.fsm.state.name == 'MINING':
                self._do_mining_routine(screen)
            reward = self.env._compute_reward()
            if self.reward_label:
                self.reward_label.setText(f"Reward: {reward:.2f}")
            time.sleep(0.2)

    def _do_mining_routine(self, screen):
        self.log("⛏ Mining routine tick")

        # 0. HOSTILE CHECK
        if self.mining.detect_hostiles(screen):
            self.log("⚠️ Hostiles detected")

        # 1. CARGO HOLD CHECK
        cargo_box = self.rh.load('mining_cargo_hold_capacity')
        if cargo_box:
            x1, y1, x2, y2 = cargo_box
            crop = screen[y1:y2, x1:x2]
            text = self.ocr.extract_text(crop)
            match = re.search(r'(\d+)', text)
            if match:
                pct = int(match.group(1))
                self.log(f"📦 Cargo: {pct}%")
                if pct >= 90:
                    self.log("🚀 Cargo full, docking...")
                    self.fsm.on_event(Event.DOCK)
                    self.mining.warp_to_station()
                    self.mining.dock_or_undock(dock=True)
                    return

        # 2. LASER MODULES
        slots = ['module_slot1', 'module_slot2', 'module_slot3']
        active = False
        for slot in slots:
            box = self.rh.load(slot)
            if box:
                x1, y1, x2, y2 = box
                if self.cv.is_module_active(screen[y1:y2, x1:x2]):
                    active = True
                    break

        if not active:
            self.log("🔄 Mining lasers inactive → activating")
            for slot in slots:
                box = self.rh.load(slot)
                if box:
                    x1, y1, x2, y2 = box
                    pyautogui.click((x1 + x2) // 2, (y1 + y2) // 2)
                    time.sleep(0.2)
            time.sleep(1.0)
            return

        # 3. TARGET LOCK
        locked = False
        box = self.rh.load("is_target_locked")
        if box:
            x1, y1, x2, y2 = box
            locked = self.cv.detect_target_lock(screen[y1:y2, x1:x2])

        if not locked:
            self.log("🔎 No target locked — acquiring new asteroid")
            self.mining.warp_to_asteroid_belt()

            # Sort overview by distance
            box = self.rh.load("overview_distance_header")
            if box:
                x1, y1, x2, y2 = box
                pyautogui.click((x1 + x2) // 2, (y1 + y2) // 2)
                self.log("↕️ Sorting overview by distance")
                time.sleep(0.5)

            # Select asteroid using OCR fallback to first row
            if self.mining.target_asteroid_via_ocr():
                self.log("🪨 Selected asteroid via OCR")
            else:
                box = self.rh.load("overview_panel")
                if box:
                    x1, y1, x2, y2 = box
                    pyautogui.click(x1 + 40, y1 + 15)
                    self.log("🪨 Selected nearest asteroid")
                    time.sleep(0.5)

            self.mining.approach_asteroid()
            self.mining.human_like_idle()


class BotGui(QtWidgets.QWidget):
    def __init__(self):
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

        self.bot = EveBot(model_path=None)
        self.bot.gui_logger = self.log_area
        self.bot.reward_label = self.reward_label

        if hasattr(self.start_btn, "clicked") and hasattr(self.start_btn.clicked, "connect"):
            self.start_btn.clicked.connect(self.bot.start)
        if hasattr(self.stop_btn, "clicked") and hasattr(self.stop_btn.clicked, "connect"):
            self.stop_btn.clicked.connect(self.bot.stop)
        if hasattr(self.override_btn, "clicked") and hasattr(self.override_btn.clicked, "connect"):
            self.override_btn.clicked.connect(self._send_manual)

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
    args = parser.parse_args()

    if args.mode == "gui":
        app = QtWidgets.QApplication(sys.argv)
        window = BotGui()
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
