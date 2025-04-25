# version: 0.1.0
# path: src/agent.py

from stable_baselines3 import PPO
import torch

class AIPilot:
    def __init__(self, env, model_path=None):
        if model_path:
            self.model = PPO.load(model_path, env=env)
        else:
            self.model = PPO('MlpPolicy', env, verbose=1)

    def pretrain(self, demo_buffer):
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

    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)
        self.model.save('ppo_trained_model')

    def decide(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
