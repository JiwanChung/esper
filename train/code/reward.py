import os
import math
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
import clip

from policy import Policy


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class SentSplit:
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

    def split_sentence(self, text):
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        if len(sentences) < 2:
            return [text]
        elif len(sentences[0]) < 10:
            return [text]
        else:
            sentences = [sentences[0], *[v for v in sentences[1:] if len(v) > 5]]
            return sentences

    def get_sentences(self, texts):
        texts = [[(i, v2) for v2 in self.split_sentence(v)] for i, v in enumerate(texts)]
        texts = list(chain(*texts))
        ids, texts = zip(*texts)
        return ids, texts

    def get_first_sentence(self, texts):
        texts = [self.split_sentence(v)[0] for i, v in enumerate(texts)]
        return texts


class Reward(nn.Module):
    def __init__(self, device, gain: float = None, bias: float = None,
                 clip_model_type: str = 'ViT-B/32',
                 clip_sent_split: str = 'none',
                 pair_reward_exp: bool = False,
                 **kwargs):
        super().__init__()

        self.device = device
        self.gain, self.bias = gain, bias
        self.clip_sent_split = clip_sent_split
        self.pair_reward_exp = pair_reward_exp
        self.norm_steps = 1000
        log.info(f'clip_sent_split: {clip_sent_split}')

        if clip_sent_split in ['first', 'mean']:
            self.sent_split = SentSplit()
        self.clip_model, self.preprocess = clip.load(clip_model_type, device=self.device, jit=False)
        for param in self.clip_model.parameters():
            param.requires_grad_(False)

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
                image_ids, input_ids, attention_mask, features, labels, _ = batch
                outputs = policy.sample(input_ids=input_ids, attention_mask=attention_mask, features=features)
                texts = outputs['response/text']
                reward = self._get_reward(features, texts)
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
                  open(Path('./') / 'reward_normalization.json', 'w'), indent=4)

    def get_reward(self, features: torch.Tensor, texts: List[str], epoch: str, device = None):
        rewards = self._get_reward(features, texts, device=device)
        return self.normalize(rewards)

    def normalize(self, rewards):
        rewards = [self.gain * x + self.bias for x in rewards]
        if self.pair_reward_exp:
            rewards = [math.exp(x / 2) - 4.48 for x in rewards]
        return rewards

    def unnormalize(self, rewards):
        if self.pair_reward_exp:
            rewards = [math.log(x + 4.48) * 2 for x in rewards]
        rewards = [(x - self.bias) / self.gain for x in rewards]
        return rewards

    def set_ln_float(self):
        for name, param in self.clip_model.named_parameters():
            if 'ln' in name:
                param.data = param.data.float()

    def _get_reward(self, features: torch.Tensor, texts: List[str], device = None):
        self.set_ln_float()
        if self.clip_sent_split == 'mean':
            reward = self._get_reward_mean(features, texts, device)
        elif self.clip_sent_split == 'first':
            reward = self._get_reward_first(features, texts, device)
        else:
            reward = self._get_reward_base(features, texts, device)
        return reward

    def _get_reward_mean(self, features: torch.Tensor, texts: List[str], device=None):
        # features: pre-extracted clip features
        if device is None:
            device = self.device
        image_features = features.to(device)
        with torch.no_grad():
            B = len(texts)
            ids, sent_texts = self.sent_split.get_sentences(texts)
            sent_text_tokens = clip.tokenize(sent_texts, truncate=True).to(device)
            text_features = self.clip_model.encode_text(sent_text_tokens)
            ids = torch.Tensor(ids).long().to(device)
            image_features = image_features.index_select(dim=0, index=ids)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            cossim = torch.einsum('bc,bc->b', image_features, text_features)

            # reward as mean of sentence cossims
            storage = torch.zeros([B]).to(device)
            for i in range(B):
                mask = (ids == i).type_as(cossim)
                if mask.sum() == 0:
                    import ipdb; ipdb.set_trace()  # XXX DEBUG
                row_mean = (cossim * mask).sum() / mask.sum()
                storage[i] = row_mean
            cossim = storage
        rewards = cossim.detach().cpu().numpy()
        return rewards

    def _get_reward_first(self, features: torch.Tensor, texts: List[str], device=None):
        # features: pre-extracted clip features
        if device is None:
            device = self.device
        image_features = features.to(device)
        with torch.no_grad():
            B = len(texts)
            sent_texts = self.sent_split.get_first_sentence(texts)
            sent_text_tokens = clip.tokenize(sent_texts, truncate=True).to(device)
            text_features = self.clip_model.encode_text(sent_text_tokens)
            image_features = image_features.index_select(dim=0, index=ids)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            cossim = torch.einsum('bc,bc->b', image_features, text_features)
        rewards = cossim.detach().cpu().numpy()
        return rewards

    def _get_reward_base(self, features: torch.Tensor, texts: List[str], device = None):
        # features: pre-extracted clip features
        if device is None:
            device = self.device
        image_features = features.to(device)
        with torch.no_grad():
            text = clip.tokenize(texts, truncate=True).to(device)
            text_features = self.clip_model.encode_text(text)
            image_features = image_features.to(text_features.dtype)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            cossim = torch.einsum('bc,bc->b', image_features, text_features)
        rewards = cossim.detach().cpu().numpy()
        return rewards
