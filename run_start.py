# run_start.py
# version: 0.3.2
# path: run_start.py

import os
import argparse
import sys
import logging
from src.logger import get_logger

# Ensure local src modules are importable when running directly
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

logger = get_logger(__name__)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.env import EveEnv


def find_latest_model(log_dir):
    """Return path to most recently saved model in a directory."""
    if not os.path.isdir(log_dir):
        return None
    candidates = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.endswith(".zip")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def make_monitored_env(log_dir):
    """Create a Monitor-wrapped EVE environment."""
    env = EveEnv()
    return Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))

def main():
    parser = argparse.ArgumentParser(description="Train or test the EVE Online bot")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test",  action="store_true", help="Run a test episode")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Number of training timesteps")
    parser.add_argument("--model_path", type=str, default="eve_bot_model", help="Path to save/load model")
    parser.add_argument("--bc_model", type=str, default=None, help="Pretrained behavior cloning model")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    # Update logger level
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    model_file = os.path.join(log_dir, args.model_path)
    if args.test and not os.path.exists(model_file):
        latest = find_latest_model(log_dir)
        if latest:
            logger.info(f"Auto-loading latest model: {latest}")
            model_file = latest
        else:
            logger.warning("No saved model found.")

    if args.train:
        # Create monitored training and eval envs
        train_env = make_monitored_env(log_dir)
        eval_env  = make_monitored_env(log_dir)

        # Checkpoint every 10k steps
        checkpoint_cb = CheckpointCallback(
            save_freq=10_000, save_path=log_dir, name_prefix="ppo_eve"
        )

        # Evaluate every 10k steps over 5 episodes
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, "best_model"),
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True
        )

        # Initialize PPO with TensorBoard logging
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=log_dir
        )
        if args.bc_model and os.path.exists(args.bc_model):
            import torch
            state = torch.load(args.bc_model, map_location="cpu")
            policy_state = model.policy.state_dict()
            for k in state:
                if k in policy_state and state[k].shape == policy_state[k].shape:
                    policy_state[k] = state[k]
            model.policy.load_state_dict(policy_state)
            logger.info(f"Loaded BC weights from {args.bc_model}")

        # Train with all callbacks
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, eval_cb],
            tb_log_name="ppo_run"
        )
        model.save(os.path.join(log_dir, args.model_path))

        logger.info(f"Training complete. Models & logs in '{log_dir}/'")

    if args.test:
        # Load model and run one test episode
        env = EveEnv()
        model = PPO.load(model_file, env=env)
        if args.bc_model and os.path.exists(args.bc_model):
            import torch
            state = torch.load(args.bc_model, map_location="cpu")
            policy_state = model.policy.state_dict()
            for k in state:
                if k in policy_state and state[k].shape == policy_state[k].shape:
                    policy_state[k] = state[k]
            model.policy.load_state_dict(policy_state)
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            logger.info(f"Action: {action}, Reward: {reward:.2f}, Done: {done}")
        logger.info("Test run complete.")

if __name__ == "__main__":
    main()
