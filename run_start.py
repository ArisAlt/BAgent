# run_start.py
# version: 0.3.1

import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.env import EveEnv

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
    args = parser.parse_args()

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

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
            print(f"Loaded BC weights from {args.bc_model}")

        # Train with all callbacks
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, eval_cb],
            tb_log_name="ppo_run"
        )
        model.save(os.path.join(log_dir, args.model_path))

        print(f"Training complete. Models & logs in '{log_dir}/'")

    if args.test:
        # Load model and run one test episode
        env = EveEnv()
        model = PPO.load(os.path.join("logs", args.model_path), env=env)
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
            print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}")
        print("Test run complete.")

if __name__ == "__main__":
    main()
