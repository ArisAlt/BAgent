# version: 0.4.2
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
from stable_baselines3 import PPO


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
    return result['data']


def record_data(filename='demo_buffer.pkl', num_samples=500, manual=True, model_path=None):
    env = EveEnv()
    demo_buffer = []
    demo_dir = os.path.join('logs', 'demonstrations')
    os.makedirs(demo_dir, exist_ok=True)
    log_path = os.path.join(demo_dir, 'log.jsonl')
    model = None
    if model_path and os.path.exists(model_path):
        model = PPO.load(model_path, env=env)

    mode = "manual" if manual else ("model" if model else "automatic")
    print(f"Starting {mode} data recording for {num_samples} samples...")
    obs = env.reset()
    stop_event = Event()

    def on_end_press(key):
        try:
            name = key.char.lower()
        except AttributeError:
            name = key.name.lower()
        if name == 'end':
            stop_event.set()
            return False

    end_listener = keyboard.Listener(on_press=on_end_press)
    end_listener.start()

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
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fp = os.path.join(demo_dir, f"{ts}.png")
            cv2.imwrite(fp, frame)
            state = env.get_observation()
            log_file.write(json.dumps({'frame': fp, 'action': label, 'state': state}) + "\n")

            obs, reward, done, info = env.step(action)
            demo_buffer.append((obs, action))
            print(f"Recorded: Step={i+1}, Action={label}, Reward={reward}")
            if done:
                obs = env.reset()

    end_listener.stop()
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
    args = parser.parse_args()

    record_data(filename=args.out, num_samples=args.samples,
                manual=args.manual, model_path=args.model)
  