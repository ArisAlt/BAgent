# version: 0.4.9
# path: data_recorder.py

import pickle
import random
import os
import json
from datetime import datetime
import cv2
from pynput import mouse, keyboard
from threading import Event
from src.env import EveEnv
from src.config import get_window_title
from stable_baselines3 import PPO
from src.logger import get_logger

logger = get_logger(__name__)


def _map_click(env, x, y):
    for idx, (typ, target) in enumerate(env.actions):
        if typ == 'click':
            coords = env.region_handler.get_coords(target)
            if coords and coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]:
                return idx, f'click_{target}'
    return None, None


def _map_key(env, key_name):
    for idx, (typ, target) in enumerate(env.actions):
        if typ == 'keypress' and key_name == target:
            return idx, f'keypress_{target}'
    return None, None


def _wait_for_event(env, stop_event=None):
    """Block until a relevant mouse or keyboard event occurs.

    If the **End** key is pressed, ``stop_event`` will be set and ``None`` is
    returned so callers can break their recording loop.
    """
    result = {}
    done = Event()

    def on_click(x, y, button, pressed):
        if pressed and not done.is_set():
            idx, label = _map_click(env, x, y)
            if idx is not None:
                result['data'] = (idx, label)
                done.set()
                return False

    def on_press(key):
        if done.is_set():
            return False
        try:
            name = key.char.lower()
        except AttributeError:
            name = key.name.lower()
        if name == 'end':
            if stop_event is not None:
                stop_event.set()
            done.set()
            return False
        idx, label = _map_key(env, name)
        if idx is not None:
            result['data'] = (idx, label)
            done.set()
            return False

    with mouse.Listener(on_click=on_click) as ml, keyboard.Listener(on_press=on_press) as kl:
        done.wait()
    ml.stop()
    kl.stop()
    return result.get('data')


def record_data(filename='demo_buffer.pkl', num_samples=500, manual=True,
                model_path=None, log_path=None, window_title=None):
    if window_title is None:
        window_title = get_window_title()
    env = EveEnv(window_title=window_title)
    demo_buffer = []
    demo_dir = os.path.join('logs', 'demonstrations')
    os.makedirs(demo_dir, exist_ok=True)
    if log_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(demo_dir, f"log_{ts}.jsonl")
    model = None
    if model_path and os.path.exists(model_path):
        model = PPO.load(model_path, env=env)

    mode = "manual" if manual else ("model" if model else "automatic")
    print(f"Starting {mode} data recording for {num_samples} samples...")
    obs = env.reset()
    stop_event = Event()

    failure_count = 0
    with open(log_path, 'a') as log_file:
        for i in range(num_samples):
            if stop_event.is_set():
                break
            if manual:
                data = _wait_for_event(env, stop_event)
                if stop_event.is_set():
                    break
                if data is None:
                    break
                idx, label = data
                action = idx
            elif model:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                label = f"model_{action}"
            else:
                action = random.randint(0, env.action_space.n - 1)
                label = f"random_{action}"

            frame = env.ui.capture()
            if frame is None:
                failure_count += 1
                logger.warning("[Recorder] Capture returned None; skipping step")
                if failure_count >= 5:
                    logger.error("[Recorder] Too many capture failures. Aborting")
                    break
                continue
            failure_count = 0

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fp = os.path.join(demo_dir, f"{ts}.png")
            cv2.imwrite(fp, frame)
            state = env.get_observation()
            log_file.write(json.dumps({'frame': fp, 'action': label, 'state': state}) + "\n")

            next_obs, reward, done, info = env.step(action)
            demo_buffer.append((state['obs'], action))
            obs = next_obs
            print(f"Recorded: Step={i+1}, Action={label}, Reward={reward}")
            if done:
                obs = env.reset()

    with open(filename, 'wb') as f:
        pickle.dump(demo_buffer, f)
    print(f"Data recording complete. Saved to {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record environment actions")
    parser.add_argument("--out", type=str, default="demo_buffer.pkl")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--log", type=str, default=None,
                        help="Path to JSONL log file")
    parser.add_argument("--window-title", type=str,
                        default=get_window_title(),
                        help="Game window title to capture")
    args = parser.parse_args()

    record_data(filename=args.out, num_samples=args.samples,
                manual=args.manual, model_path=args.model,
                log_path=args.log, window_title=args.window_title)
  