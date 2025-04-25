# version: 0.3.0
# path: src/bot_core.py

from src.env import EveEnv
from src.agent import AIPilot
from src.ui import Ui
import time

class EveBot:
    def __init__(self, model_path=None):
        self.env = EveEnv(max_actions=20)
        self.agent = AIPilot(model_path=model_path)
        self.ui = Ui()

    def run(self):
        """
        Run the bot loop: capture, process, decide, and act.
        """
        obs = self.env.reset()
        done = False
        while not done:
            action = self.agent.decide(obs)  # Get action from the trained agent
            obs, reward, done, info = self.env.step(action)  # Perform action
            print(f"Action: {action}, Reward: {reward}, Done: {done}")
            time.sleep(1)  # Add delay to simulate human-like behavior (optional)

if __name__ == "__main__":
    bot = EveBot(model_path="eve_bot_model.zip")  # Path to the trained model
    bot.run()
