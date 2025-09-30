# EVE Online Bot Project Scaffold
# version: 0.8.1
# path: Scaffold.md

---

## Directory Structure
```
BAgent/
├── README.md           # version: 0.7.2 | path: README.md
├── src/
│   ├── __init__.py       # version: 0.1.0 | path: src/__init__.py
│   ├── bot_core.py       # version: 0.9.0 | path: src/bot_core.py
│   ├── env.py            # version: 0.5.0 | path: src/env.py
│   ├── agent.py          # version: 0.5.3 | path: src/agent.py
│   ├── ocr.py            # version: 0.3.7 | path: src/ocr.py
│   ├── cv.py             # version: 0.4.0 | path: src/cv.py
│   ├── detector.py       # version: 0.1.0 | path: src/detector.py
│   ├── ui.py             # version: 0.6.0 | path: src/ui.py
│   ├── capture_utils.py  # version: 0.8.5 | path: src/capture_utils.py
│   ├── logger.py         # version: 0.1.0 | path: src/logger.py
│   ├── roi_capture.py    # version: 0.2.5 | path: src/roi_capture.py
│   ├── mining_actions.py # version: 0.2.0 | path: src/mining_actions.py
│   ├── ocr_finetune.py   # version: 0.1.0 | path: src/ocr_finetune.py
│   ├── roi_live_overlay.py # version: 0.3.1 | path: src/roi_live_overlay.py
│   ├── state_machine.py  # version: 0.2.0 | path: src/state_machine.py
│   ├── llm_client.py     # version: 0.2.0 | path: src/llm_client.py
│   ├── config/
│   │   ├── agent_config.yaml # version: 0.3.0 | path: src/config/agent_config.yaml
│   │   └── pilot_name.txt    # version: 0.1.0 | path: src/config/pilot_name.txt
│   └── roi_screenshots/  # ROI screenshot samples
├── env.py               # version: 0.1.0 | path: env.py
├── roi_capture.py       # version: 0.1.2 | path: roi_capture.py
├── bot_core.py          # version: 0.1.0 | path: bot_core.py
#   └─ thin wrappers re-exporting the real modules under src/
├── run_start.py          # version: 0.3.3 | path: run_start.py
├── data_recorder.py      # version: 0.4.8 | path: data_recorder.py
├── export_ocr_samples.py # version: 0.1.3 | path: export_ocr_samples.py
├── generate_box_files.py # version: 0.1.1 | path: generate_box_files.py
├── pre_train_data.py     # version: 0.3.0 | path: pre_train_data.py
├── replay_session.py     # version: 0.3.0 | path: replay_session.py
├── replay_correction.py     # version: 0.1.0 | path: replay_correction.py
├── search.py            # version: 0.1.0 | path: search.py
├── tag.py               # version: 0.1.0 | path: tag.py
├── add_tesseract_to_path.bat # helper script to set PATH on Windows
├── ets.txt               # sample training commands
├── promts.txt            # project prompts and notes
├── regions.yaml          # saved ROI definitions
├── requirements.txt      # version: 0.3.1 | path: requirements.txt
├── test_env.py           # version: 0.2.0 | path: test_env.py
├── tests/                # test suite
│   ├── test_capture_utils.py      # version: 0.1.0 | path: tests/test_capture_utils.py
│   ├── test_env_actions.py        # version: 0.2.0 | path: tests/test_env_actions.py
│   ├── test_gui_cli_integration.py # version: 0.1.0 | path: tests/test_gui_cli_integration.py
│   ├── test_region_handler.py     # version: 0.1.0 | path: tests/test_region_handler.py
│   └── test_replay_session.py     # version: 0.1.0 | path: tests/test_replay_session.py
├── sitecustomize.py      # version: 0.1.0 | path: sitecustomize.py
├── training_texts_dir/   # OCR training data


---

## Recent Changes Summary

- **Log Tooling:**
  - Added `search.py` for structured searching across JSONL demonstration logs with
    substring, regex, or callable predicates and optional nested field selectors.
  - Introduced `tag.py` with a `TagStore` utility to normalise, deduplicate, and persist
    metadata tags for recorded sessions.

---

## Next Steps

1. **Capture & Validate ROIs** for all new regions using `roi_capture.py`.
2. **Detector Labeling**: capture or update `detect`-type ROIs with the
   desired YOLO class labels and ensure `detector.class_names` covers them.
3. **Reward Tuning & Logging**: refine `_compute_reward` weights and add metrics.
4. **Complete Agent Module** with decision logic and automated action recording.
5. **Expand Gym Environment** for complex mission scenarios.
6. **Full Integration Testing** of ROI/UI functionality via GUI and CLI.
7. **Fine-tune OCR** with `ocr_finetune.py`.
8. **Document Windows Setup**: run `add_tesseract_to_path.bat` to add Tesseract to `PATH`.
9. **Update Tests & Integration** to cover new modules.
10. **Documentation & Packaging**: finalize README, version bump, and release.

---

## Requirements

```txt
paddleocr
opencv-python
PySide6
numpy
pillow
pyautogui
gym
stable-baselines3
torch
pyyaml
pynput
scikit-learn
onnxruntime
mss
pywin32 (Windows only)
requests
```


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
  - `capture_screen` now falls back to standard screenshot methods if
    `PrintWindow` fails, logging which method succeeded.
  - ROI capture and validation logic moved to `roi_capture.py`.
- **Data Recording & Pretraining**:
- `data_recorder.py` logs frame screenshots, observations and semantic actions.
- Recording skips steps when `env.ui.capture()` returns `None` and aborts after
  repeated failures.
- Recording can be terminated early with the **End** key.
  - The pickled buffer now stores the observation before the action is executed.
  - Scripts for behavior cloning from recorded data.
  - `agent.py` includes BC training and inference helpers.
  - `bot_core.py` central bot loop connecting all modules.
  - `test_gui_cli_integration.py` added for GUI and CLI integration testing.

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




