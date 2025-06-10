# version: 0.4.2
# path: src/bot_core.py

import sys
import threading
import time
import logging
from env import EveEnv
from agent import AIPilot
from ui import Ui
from state_machine import FSM, Event, State
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QPushButton, QLineEdit, QTextEdit,
    QVBoxLayout, QLabel
)
from PySide6.QtCore import Qt


class EveBot:
    def __init__(self, model_path=None):
        # Initialize environment, agent, UI, and FSM
        self.env = EveEnv()
        self.agent = AIPilot(model_path=model_path)
        self.ui = Ui()
        self.fsm = FSM()
        self.running = False

    def run(self):
        """
        Main loop: capture UI, dispatch FSM events, decide and execute actions.
        """
        logger = logging.getLogger('EveBot')
        self.running = True

        # Reset environment to get initial observation
        obs = self.env.reset()
        done = False

        while self.running and not done:
            # 1. Capture screen and extract data
            screenshot = self.ui.capture()
            text_data = self.env.ocr.extract_text(screenshot).lower()
            elements = self.env.cv.detect_elements(
                screenshot,
                templates=self.env._load_templates()
            )

            # 2. Dispatch FSM events based on UI cues
            self.dispatch_events(elements, text_data)
            logger.info(f"FSM State: {self.fsm.state.name}")

            # 3. Decide next action based on current FSM state
            action = self.decide_action(obs)

            # 4. Execute action and step environment
            obs, reward, done, info = self.env.step(action)

            # 5. Log details
            logger.info(
                f"Action: {action}, Reward: {reward:.2f}, Done: {done}, New State: {self.fsm.state.name}"
            )

            # Small delay to control loop speed
            time.sleep(1)

        logger.info("Bot run loop exited.")

    def stop(self):
        """Stop the bot loop."""
        self.running = False

    def dispatch_events(self, elements, text_data):
        """
        Map detected UI elements and OCR text to FSM events.
        """
        # Mining start detection
        if any(e['name'] == 'mining_laser_on' for e in elements):
            self.fsm.on_event(Event.START_MINING)

        # Hostile encountered
        if any(e['name'] == 'hostile_alert' for e in elements):
            self.fsm.on_event(Event.ENEMY_DETECTED)

        # Dock/Undock
        if 'docked' in text_data:
            self.fsm.on_event(Event.DOCK)
        elif 'undocked' in text_data:
            self.fsm.on_event(Event.UNDOCK)

    def decide_action(self, obs):
        """
        Choose action based on current FSM state.
        Extend this with state-specific logic as needed.
        """
        state = self.fsm.state
        if state == State.MINING:
            # TODO: implement mining-specific actions
            return self.agent.decide(obs)
        elif state == State.COMBAT:
            # TODO: implement combat-specific actions
            return self.agent.decide(obs)
        elif state == State.DOCKING:
            # TODO: implement docking-specific actions
            return self.agent.decide(obs)
        else:
            # IDLE, MISSION, EXPLORATION, etc.
            return self.agent.decide(obs)


class QTextEditLogger(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        msg = self.format(record)
        self.text_edit.append(msg)


class BotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EVE Bot Control")
        self.bot = None
        self.thread = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Model path input
        self.model_input = QLineEdit("eve_bot_model.zip")
        layout.addWidget(QLabel("Model Path:"))
        layout.addWidget(self.model_input)

        # Start/Stop buttons
        self.start_button = QPushButton("Start Bot")
        self.start_button.clicked.connect(self.start_bot)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Bot")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_bot)
        layout.addWidget(self.stop_button)

        # Log display
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        # Configure logging to GUI
        logger = logging.getLogger('EveBot')
        logger.setLevel(logging.INFO)
        handler = QTextEditLogger(self.log_area)
        handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%H:%M:%S'))
        logger.addHandler(handler)
        logger.info("GUI Ready.")

    def start_bot(self):
        model_path = self.model_input.text().strip()
        self.bot = EveBot(model_path=model_path)
        self.thread = threading.Thread(target=self.bot.run, daemon=True)
        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        logging.getLogger('EveBot').info("Bot started.")

    def stop_bot(self):
        if self.bot:
            self.bot.stop()
            self.thread.join()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            logging.getLogger('EveBot').info("Bot stopped by user.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = BotGUI()
    gui.resize(600, 400)
    gui.show()
    sys.exit(app.exec())
