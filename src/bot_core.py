# version: 0.5.0
# path: src/bot_core.py

import sys
import time
import re
import pyautogui
import cv2
from PySide6 import QtWidgets, QtCore, QtGui

from roi_capture import RegionHandler
from ocr import OcrEngine
from cv import CvEngine
from state_machine import FSM, Event

class EveBot:
<<<<<<< HEAD
    def __init__(self, model_path=None):
        # Initialize environment, agent, UI, and FSM
        self.env = EveEnv()
        self.agent = AIPilot(model_path=model_path)
        self.ui = Ui()
        self.fsm = FSM()
=======
    def __init__(self, config):
        self.config = config
>>>>>>> origin/Save
        self.running = False
        self.fsm = FSM()
        self.rh = RegionHandler()
        self.ocr = OcrEngine()
        self.cv = CvEngine()
        self.gui_logger = None

    def log(self, message):
        timestamped = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(timestamped)
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

    def _main_loop(self):
        import capture_utils
        while self.running:
            screen = capture_utils.capture_screen(select_region=False)
            if self.fsm.state.name == 'MINING':
                self._do_mining_routine(screen)
            time.sleep(0.2)

    def _do_mining_routine(self, screen):
        self.log("⛏ Mining routine tick")

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

            # Sort overview by distance
            box = self.rh.load("overview_distance_header")
            if box:
                x1, y1, x2, y2 = box
                pyautogui.click((x1 + x2) // 2, (y1 + y2) // 2)
                self.log("↕️ Sorting overview by distance")
                time.sleep(0.5)

            # Click first asteroid row
            box = self.rh.load("overview_panel")
            if box:
                x1, y1, x2, y2 = box
                pyautogui.click(x1 + 40, y1 + 15)
                self.log("🪨 Selected nearest asteroid")
                time.sleep(0.5)

            # Click Approach
            box = self.rh.load("approach_button")
            if box:
                x1, y1, x2, y2 = box
                pyautogui.click((x1 + x2) // 2, (y1 + y2) // 2)
                self.log("🛸 Approaching target")
                time.sleep(0.5)


class BotGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EVE Bot Controller")
        layout = QtWidgets.QVBoxLayout(self)

        self.log_area = QtWidgets.QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        self.start_btn = QtWidgets.QPushButton("Start Bot")
        self.stop_btn = QtWidgets.QPushButton("Stop Bot")
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        self.bot = EveBot(config={})
        self.bot.gui_logger = self.log_area

        self.start_btn.clicked.connect(self.bot.start)
        self.stop_btn.clicked.connect(self.bot.stop)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = BotGui()
    window.show()
    sys.exit(app.exec())
