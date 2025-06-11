# version: 0.1.0
# path: replay_correction.py

import argparse
import json
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from src.env import EveEnv
from pre_train_data import BCModel
from stable_baselines3 import PPO


def _label_from_idx(env: EveEnv, idx: int) -> str:
    typ, target = env.actions[idx]
    if typ == "click":
        return f"click_{target}"
    if typ == "keypress":
        return f"keypress_{target}"
    return "sleep"


def _label_to_idx(env: EveEnv, label: str) -> Optional[int]:
    for idx, (typ, target) in enumerate(env.actions):
        if _label_from_idx(env, idx) == label:
            return idx
    if "_" in label:
        try:
            return int(label.split("_")[-1])
        except ValueError:
            return None
    return None


def load_model(path: str, env: EveEnv) -> Tuple[object, str]:
    if path.endswith(".pt"):
        model = BCModel(env.observation_space.shape[0], env.action_space.n)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, "bc"
    model = PPO.load(path, env=env)
    return model, "ppo"


def _predict(model, model_type: str, obs: np.ndarray) -> Tuple[int, float]:
    if model_type == "bc":
        with torch.no_grad():
            tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            idx = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0, idx].item())
            return idx, conf
    idx, _ = model.predict(obs, deterministic=True)
    return int(idx), 1.0


def correct_log(log_file: str, out_file: str, delay: int = 0,
                model_path: Optional[str] = None):
    env = EveEnv()
    model = None
    model_type = ""
    if model_path:
        model, model_type = load_model(model_path, env)

    with open(log_file, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    idx = 0
    while idx < len(entries):
        entry = entries[idx]
        frame = cv2.imread(entry.get("frame", ""))
        if frame is None:
            idx += 1
            continue
        label = entry.get("action", "")
        state = entry.get("state", {})
        pred_label = ""
        if model and "obs" in state:
            pred_idx, conf = _predict(
                model, model_type, np.array(state["obs"], dtype=np.float32)
            )
            pred_label = f"Pred: {_label_from_idx(env, pred_idx)} ({conf:.2f})"
        y = 20
        cv2.putText(frame, f"Actual: {label}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if pred_label:
            y += 20
            cv2.putText(frame, pred_label, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y += 20
        cv2.putText(frame, "[0-9]: correct action, space: next, q: quit", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Correction", frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        if ord('0') <= key <= ord('9'):
            a_idx = key - ord('0')
            if a_idx < env.action_space.n:
                entry['action'] = _label_from_idx(env, a_idx)
                entry['weight'] = 2.0
                entry['corrected'] = True
        idx += 1
    cv2.destroyAllWindows()

    with open(out_file, 'w') as f:
        for e in entries:
            json.dump(e, f)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Correct recorded session logs")
    parser.add_argument("--log", type=str, required=True,
                        help="Path to original log file")
    parser.add_argument("--out", type=str, required=True,
                        help="Path to save corrected log")
    parser.add_argument("--delay", type=int, default=0,
                        help="Delay for cv2.waitKey")
    parser.add_argument("--model", type=str, default=None,
                        help="Optional PPO or BC model path")
    args = parser.parse_args()
    correct_log(args.log, args.out, args.delay, args.model)


if __name__ == "__main__":
    main()
