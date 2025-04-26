# EVE Online Bot Project Scaffold

> version: 0.3.0\
> updated: Added demo AI pilot implementation (Gym env & agent), mining actions

---

## Directory Structure
```
EVEBot/
├── src/
│   ├── bot_core.py       # version: 0.3.0 | path: src/bot_core.py
│   ├── env.py            # version: 0.3.0 | path: src/env.py
│   ├── agent.py          # version: 0.1.0 | path: src/agent.py
│   ├── ocr.py            # version: 0.3.2 | path: src/ocr.py
│   ├── cv.py             # version: 0.3.2 | path: src/cv.py
│   ├── ui.py             # version: 0.3.6 | path: src/ui.py
│   ├── capture_utils.py  # version: 0.1.0 | path: src/capture_utils.py
│   └── roi_capture.py    # version: 0.1.6 | path: src/roi_capture.py
├── run_start.py          # version: 0.2.0 | path: run_start.py
├── data_recorder.py      # version: 0.3.0 | path: data_recorder.py
├── pretrain_model.py     # version: 0.1.0 | path: pretrain_model.py
├── requirements.txt      # version: 0.4.0 | path: requirements.txt
├── README.md             # version: 0.4.1 | path: README.md

```

## Demo AI Pilot Code

### src/env.py

```python
# version: 0.3.0
# path: src/env.py

import gym
from gym import spaces
import numpy as np
from ocr import OcrEngine
from cv import CvEngine
from ui import Ui

class EveEnv(gym.Env):
    """
    OpenAI Gym wrapper for EVE UI as state-action interface.
    State: concatenated OCR text embeddings + element positions.
    Action: discrete commands mapped to UI actions.
    """
    def __init__(self, max_actions=20):
        super().__init__()
        self.ocr = OcrEngine()
        self.cv = CvEngine()
        self.ui = Ui()

        # Example: 100-dimensional state vector
        obs_dim = 100
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Discrete action space: click hotspots or key presses
        self.action_space = spaces.Discrete(max_actions)

    def reset(self):
        # Optionally navigate to a start location in-game
        obs = self._get_obs()
        return obs

    def step(self, action):
        # Map discrete action to UI command
        cmd = self._action_to_command(action)
        self.ui.execute(cmd)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}
        return obs, reward, done, info

    def _get_obs(self):
        img = self.ui.capture()
        text = self.ocr.extract_text(img)
        elems = self.cv.detect_elements(img)
        # Feature engineering: embed text + flatten element coords
        # (Here: dummy zero vector)
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _action_to_command(self, action):
        # Define mapping e.g., action 0=click at slot1, 1=press f1, ...
        # Return dict like {'type':'click','x':100,'y':200}
        return {'type': 'no-op'}

    def _compute_reward(self):
        # Example placeholder: reward by positive in-game events
        return 0.0

    def _check_done(self):
        # End of episode logic
        return False
```

### src/agent.py

```python
# version: 0.3.0
# path: src/agent.py

import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class AIPilot:
    def __init__(self, model_path=None):
        self.env = gym.make('EveEnv-v0')  # register below
        if model_path:
            self.model = PPO.load(model_path, env=self.env)
        else:
            # Initialize a new model with MLP policy
            self.model = PPO('MlpPolicy', self.env, verbose=1)

    def pretrain(self, demo_buffer):
        """
        Behavior Cloning on demonstration data.
        demo_buffer: list of (obs, action).
        """
        obs, acts = zip(*demo_buffer)
        obs = torch.tensor(obs, dtype=torch.float32)
        acts = torch.tensor(acts, dtype=torch.long)
        optimizer = torch.optim.Adam(self.model.policy.parameters(), lr=1e-4)
        for epoch in range(10):
            dist = self.model.policy.get_distribution(obs)
            loss = -dist.log_prob(acts).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.model.save('bc_pretrained')

    def train(self, timesteps=1_000_00):
        checkpoint = CheckpointCallback(save_freq=10_000, save_path='./', name_prefix='ppo_eve')
        self.model.learn(total_timesteps=timesteps, callback=checkpoint)
        self.model.save('ppo_eve_final')

    def decide(self, text_data, elements):
        obs = self.env._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        cmd = self.env._action_to_command(action)
        return cmd
```

### src/bot_core.py

```python
# version: 0.3.0
# path: src/bot_core.py

from agent import AIPilot

class EveBot:
    def __init__(self, agent=None):
        from ocr import OcrEngine
        from cv import CvEngine
        from ui import Ui

        self.ocr = OcrEngine()
        self.cv = CvEngine()
        self.ui = Ui()
        self.agent = agent or AIPilot()

    def run(self):
        while True:
            screenshot = self.ui.capture()
            text_data = self.ocr.extract_text(screenshot)
            elements = self.cv.detect_elements(screenshot)
            action = self.agent.decide(text_data, elements)
            self.ui.execute(action)
```

---

## Feature List

- **Automated Resource Gathering**: mining, salvaging, hauling.
- **Market Trading Assistant**: price checks, buy/sell order placement, profit calculation.
- **Mission Runner Navigation**: auto-warp, bookmark management, target selection.
- **Combat Support**: target prioritization, module cycling, turret/launcher control.
- **Fleet Coordination**: position broadcasting, fleet channel messages, bookmark sharing.
- **Skill Training Monitor**: track skill queue, recommend injectors or daily planning.
- **Dynamic Macros & Scripting**: user-defined chains of actions with parameters.
- **AI-based Decision-making**: route optimization, adaptive reaction to threats.
- **Real-time Alerts & Notifications**: desktop pop-ups, Discord/webhook integration.
- **Headless/CLI Mode**: run without UI with config files; manual override interface.

---

## Recommended Mining Actions

1. **Warp to asteroid belt**
2. **Approach asteroid**
3. **Activate mining lasers**
4. **Monitor cycle completion**
5. **Check cargo hold**
6. **Recover ore fragments**
7. **Filter & jettison unwanted items**
8. **Randomize camera movement**
9. **Adjust overview filters**
10. **Detect hostiles**
11. **Warp to station/base**
12. **Dock/undock sequence**
13. **Refine or sell ore**
14. **Log statistics & events**
15. **Human-like idle actions**

*Include randomized delays (±5–15%), slight coordinate offsets, viewport pans, and breaks.*

---

## AI Pilot Implementation Recommendation

1. Use RL/IL (Stable Baselines3 PPO) with a Gym env wrapper.
2. Record demos for Behavior Cloning pre-training.
3. Fine-tune with RL: reward=ISK/hr/ship integrity.
4. Export PyTorch model; integrate via `AIPilot.decide()`.
5. Monitor metrics and provide manual override UI.

---

## Recent Changes Summary

- **UI Module Enhancements**:
  - Randomized delays and jitter for human-like behavior.
  - Integrated `RegionHandler` for ROI management.
- **ROI Capture Utilities**:
  - Persistent preview saving.
  - Input validation for coordinates and region names.
  - Backup handling for YAML files.
  - List and delete ROI functionality.
- **Modularization**:
  - Screen capture separated into `capture_utils.py`.
  - ROI capture and validation logic moved to `roi_capture.py`.
- **Data Recording & Pretraining**:
  - `data_recorder.py` allows manual/automatic action logging.
  - `pretrain_model.py` for behavior cloning from recorded data.
  - Placeholder `agent.py` for PPO model management.
  - `bot_core.py` central bot loop connecting all modules.

---

## OpenAI Gym Wrapper for EVE UI

- **Environment**: `EveEnv` wraps EVE Online's UI as a Gym-compatible environment.
- **Observation Space**: Combination of OCR-extracted text embeddings and CV-detected element positions.
- **Action Space**: Discrete actions mapped to UI commands (clicks, keypresses).
- **Rewards**: Defined by ISK gains, successful actions, and safety metrics.
- **Episodes**: Structured around task completion, e.g., mining cycles or combat engagements.
- **Training Flow**:
  - Behavior Cloning with recorded demonstrations.
  - PPO Fine-tuning using stable-baselines3.
  - Model export for inference-driven bot control.

---

## Next Steps

- Integration testing of new ROI and UI functionality.
- Complete agent module with decision logic.
- Expand Gym environment for complex mission scenarios.
- Automate model loading and smart action recording.
---

## Requirements

```txt
pytesseract
paddleocr
opencv-python
torchvision
PySide6
numpy
pillow
pyautogui
gym
stable-baselines3
torch
```

