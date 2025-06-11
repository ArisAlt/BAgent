import argparse
import json
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from pre_train_data import BCModel
from src.env import EveEnv
from stable_baselines3 import PPO


def _label_from_idx(env: EveEnv, idx: int) -> str:
    """Return the semantic label for a given action index."""
    typ, target = env.actions[idx]
    if typ == "click":
        return f"click_{target}"
    if typ == "keypress":
        return f"keypress_{target}"
    return "sleep"


def _label_to_idx(env: EveEnv, label: str) -> Optional[int]:
    """Map a log label back to an action index."""
    for idx, (typ, target) in enumerate(env.actions):
        l = _label_from_idx(env, idx)
        if label == l:
            return idx
    if "_" in label:
        try:
            return int(label.split("_")[-1])
        except ValueError:
            return None
    return None


def load_model(path: str, env: EveEnv) -> Tuple[object, str]:
    """Load a PPO or BC model depending on file extension."""
    if path.endswith(".pt"):
        model = BCModel(env.observation_space.shape[0], env.action_space.n)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, "bc"
    model = PPO.load(path, env=env)
    return model, "ppo"


def _predict(model, model_type: str, obs: np.ndarray) -> Tuple[int, float]:
    """Return predicted action index and confidence."""
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


def replay(log_file: str, delay: int = 500, model_path: Optional[str] = None,
           accuracy_out: Optional[str] = None):
    """Display frames from a recorded session with optional model comparison."""
    env = EveEnv()
    model = None
    model_type = ""
    if model_path:
        model, model_type = load_model(model_path, env)

    correct = 0
    total = 0

    with open(log_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            frame = cv2.imread(entry["frame"])
            if frame is None:
                continue
            label = entry.get("action", "")
            state = entry.get("state", {})

            y = 20
            cv2.putText(frame, f"Actual: {label}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if model and "obs" in state:
                pred_idx, conf = _predict(model, model_type,
                                          np.array(state["obs"],
                                                   dtype=np.float32))
                pred_label = _label_from_idx(env, pred_idx)
                total += 1
                actual_idx = _label_to_idx(env, label)
                if actual_idx is not None and actual_idx == pred_idx:
                    correct += 1
                    color = (0, 200, 0)
                else:
                    color = (0, 0, 255)
                y += 20
                msg = f"Pred: {pred_label} ({conf:.2f})"
                cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 1)

            for k, v in state.items():
                if k == "obs":
                    continue
                y += 20
                cv2.putText(frame, f"{k}: {v}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow("Replay", frame)
            key = cv2.waitKey(delay)
            if key & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

    if model and accuracy_out:
        acc = correct / total if total else 0.0
        with open(accuracy_out, "w") as f_out:
            json.dump({"accuracy": acc}, f_out)
        print(f"Accuracy: {acc:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Replay recorded session")
    parser.add_argument("--log", type=str, default="recordings/log.jsonl",
                        help="Path to log file")
    parser.add_argument("--delay", type=int, default=500,
                        help="Delay between frames in ms")
    parser.add_argument("--model", type=str, default=None,
                        help="Optional path to PPO or BC model")
    parser.add_argument("--accuracy-out", type=str, default=None,
                        help="Write model accuracy to this file")
    args = parser.parse_args()
    replay(args.log, args.delay, args.model, args.accuracy_out)


if __name__ == "__main__":
    main()
