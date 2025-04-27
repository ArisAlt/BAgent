# version: 0.4.0
# path: src/bot_core.py

import sys
import threading
import time
import logging
from src.env import EveEnv
from src.agent import AIPilot
from src.ui import Ui
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QPushButton, QLineEdit, QTextEdit,
    QVBoxLayout, QLabel
)
from PySide6.QtCore import Qt

class EveBot:
    def __init__(self, model_path=None):
        self.env = EveEnv(max_actions=20)
        self.agent = AIPilot(model_path=model_path)
        self.ui = Ui()
        self.running = False

    def run(self):
        self.running = True
        obs = self.env.reset()
        done = False
        while self.running and not done:
            action = self.agent.decide(obs)
            obs, reward, done, info = self.env.step(action)
            logging.getLogger('EveBot').info(f"Action: {action}, Reward: {reward}, Done: {done}")
            time.sleep(1)
        logging.getLogger('EveBot').info("Bot stopped.")

    def stop(self):
        self.running = False

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

        self.model_input = QLineEdit("eve_bot_model.zip")
        layout.addWidget(QLabel("Model Path:"))
        layout.addWidget(self.model_input)

        self.start_button = QPushButton("Start Bot")
        self.start_button.clicked.connect(self.start_bot)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Bot")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_bot)
        layout.addWidget(self.stop_button)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        # Setup logging
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
