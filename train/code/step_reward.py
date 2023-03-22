from typing import List

import torch

from .reward import Reward


class StepReward(Reward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_steps(self, lm, text, device):
        ids = self.tokenizer.encode(text)
        storage = []
        for i in len(1, ids):
            step = ids[:i]

    def _get_reward(self, lm, features: torch.Tensor, texts: List[str], device = None):
        with torch.no_grad():
            for i in range(len(texts)):
                steps = self.get_steps(lm, texts[i], device)
                feat = features[i].unsqueeze(0)
                rewards = super()._get_reward(feat, steps, device=device)
        return rewards
