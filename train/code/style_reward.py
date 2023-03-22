import os
import json
import logging
from itertools import chain
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from policy import Policy
from bert_scorer import BertScorer


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class StyleReward(nn.Module):
    def __init__(self, device, gain: float = None, bias: float = None,
                 style_cache_path: str = '', label_path: str = '',
                 style_acc_as_rewards: bool = False):
        super().__init__()

        self.device = device
        self.gain, self.bias = gain, bias
        self.norm_steps = 100
        self.style_acc_as_rewards = style_acc_as_rewards
        if self.style_acc_as_rewards:
            log.info("using style accuracy as rewards")

        self.bert = BertScorer(style_cache_path, label_path, device)

    def set_reward_norm(self, dataloader: DataLoader, policy: Policy,
                        new_mean: int = 0., new_std: int = 1.):
        # normalize the rewards to have mean 0, var 1
        if self.gain is None and self.bias is None:
            log.info('compute reward statistics before normalization ...')
        else:
            log.info(f'reward after normalization: mean={new_mean}, std={new_std}')
            log.info(f'normalization factor: gain={self.gain}, bias={self.bias}')
            return

        rewards = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
                image_ids, input_ids, attention_mask, features, labels = batch
                outputs = policy.sample(input_ids=input_ids, attention_mask=attention_mask, features=features)
                texts = outputs['response/text']
                reward, _ = self._get_reward(labels, texts)
                rewards = [*rewards, *reward]
                if i >= self.norm_steps:
                    break

        old_mean, old_std = np.mean(rewards), np.std(rewards)
        log.info('statistics:')
        log.info(f'reward before normalization: mean={old_mean}, std={old_std}')

        # gain * N(old_mean,old_std) + bias = N(gain * old_mean, gain * old_std) + bias
        #                                   = N(gain * old_mean + bias, gain * old_std)
        # gain * old_std = new_std, gain = new_std / old_std
        # gain * old_mean + bias = new_mean, bias = new_mean - gain * old_mean
        self.gain = new_std / old_std
        self.bias = new_mean - self.gain * old_mean
        log.info(f'reward after normalization: mean={new_mean}, std={new_std}')
        log.info(f'normalization factor: gain={self.gain}, bias={self.bias}')

        json.dump({'old_mean': float(old_mean), 'old_std': float(old_std),
                   'new_mean': float(new_mean), 'new_std': float(new_std),
                   'gain': float(self.gain), 'bias': float(self.bias)},
                  open(Path('./') / 'style_reward_normalization.json', 'w'), indent=4)

    def get_reward(self, labels: torch.Tensor, texts: List[str], epoch: str, device = None):
        rewards, acc = self._get_reward(labels, texts, device=device)
        return self.normalize(rewards), acc

    def normalize(self, rewards):
        return [self.gain * x + self.bias for x in rewards]

    def unnormalize(self, rewards):
        return [(x - self.bias) / self.gain for x in rewards]

    def _get_reward(self, labels: torch.Tensor, texts: List[str], device = None):
        # labels: style category indices
        if device is None:
            device = self.device
        labels = labels.to(device)
        rewards, acc = self.bert(texts, labels, device=device)
        if self.style_acc_as_rewards:
            rewards = acc
        return rewards, acc
