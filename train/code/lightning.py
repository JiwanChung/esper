import os
import math
import traceback
import json
import logging
import time
import random
import argparse
import itertools
import types
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, Any

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.upgrade_checkpoint import KEYS_MAPPING as DEPRECATED_CHECKPOINT_KEYS
from pytorch_lightning.strategies.ddp import DDPStrategy

from arguments import get_args
from data import (
    ClipCocoCollator, ConcatDataset,
    ClipCocoDataset, ClipOpenImagesDataset, ClipGoodNewsDataset
)
from policy import Policy
from ref_policy import RefPolicy
from value import Value
from reward import Reward
from style_reward import StyleReward
from utils.utils import (
    ensure_dir, ceil_div, exact_div,
    whiten, reduce_mean, reduce_sum, reduce_std,
    clamp, flatten_dict,
    get_first_sentence, get_longest,
    get_jsd, get_kl, get_mean_kl
)
from metric.lang_metrics import Eval
from load import find_last_checkpoint, download_weights, load_finetuned
from replay import Buffer, count_rep
from loss.clip import get_clip_loss
from loss.repetition import (
    get_gpt_repetition_penalty,
    get_whitespace_repetition_penalty
)


from deepspeed.utils.logging import logger, get_current_level
# ignore deprecation warning from deepspeed
# pytorch lightning needs it as of new
logger.setLevel(logging.ERROR)
logger.handlers[0].setLevel(logging.ERROR)


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class DumpCallback(Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.text_data = []

    def on_validation_epoch_end(self, trainer, pl_module):
        data = pl_module.text_data

        if pl_module.hparams.use_coco_eval:
            hypos = defaultdict(lambda : [])
            tgts = defaultdict(lambda : [])
            for row in data:
                hypos[row['image_id']].append(row['hypo'])
                tgts[row['image_id']].extend(row['tgt'])
            hypos = {k: get_longest(v) for k, v in hypos.items()}
            tgts = {k: list(set(v)) for k, v in tgts.items()}
            keys = list(tgts.keys())
            hypos = [hypos[k] for k in keys]
            tgts = [tgts[k] for k in keys]
            batch_size = len(hypos)

            metrics = pl_module.metrics(hypos, tgts)
            print('------')
            print('epoch metrics')
            for k, v in metrics.items():
                print(f"  {k} = {v:+.2f}")
                pl_module.log(f'Evaluation/{k}', v, on_epoch=True, batch_size=batch_size)
            print('------')

        log_dir = pl_module.hparams.save_dir
        coco_dir = Path(log_dir) / 'coco_eval'
        coco_dir.mkdir(exist_ok=True)

        with open(coco_dir / f'step_{pl_module.global_step}.json', 'w') as f:
            json.dump(data, f, indent=4)


class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class AdaptiveKLController:
    def __init__(self, init_kl_coef, params):
        self.value = init_kl_coef
        self.hparams = params

    def update(self, current, n_steps):
        target = self.hparams.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult


class PPOTrainer(pl.LightningModule):
    def __init__(self, **params):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False  # optimize in training_step

    def get_global_step(self):
        # global number of calls for train_step. This takes care of multiple updates and multi gpus
        return self.trainer.fit_loop.total_batch_idx
        '''
        # not anymore
        if self.hparams.num_gpus < 2:
            return self.global_step  # this is sufficient for single-gpu case, weirdly though.
        return self.global_step // (self.hparams.nminibatches * self.hparams.num_gpus)
        '''

    @property
    def using_deepspeed(self):
        return self.hparams.use_deepspeed and self.hparams.num_gpus > 1

    @property
    def use_ref_policy(self):
        return self.hparams.use_ref_gen_until is not None and self.hparams.use_ref_gen_until > self.get_global_step()

    @property
    def use_only_kl(self):
        return self.hparams.use_only_kl_until is not None and self.hparams.use_only_kl_until > self.get_global_step()

    @property
    def use_goal_scores(self):
        return self.hparams.do_not_use_goal_before is None or self.hparams.do_not_use_goal_before <= self.get_global_step()

    def setup(self, stage: str = None):
        train_dataset = ClipCocoDataset(model_name=self.hparams.init_model,
                                        split='train', clip_model_type=self.hparams.clip_model_type,
                                        use_caption=self.hparams.use_caption,
                                        label_path=self.hparams.label_path,
                                        sent_init_path=self.hparams.sent_init_path,
                                        fixed_prompt=self.hparams.fixed_prompt,
                                        eval_label=self.hparams.eval_label,
                                        use_coco_eval=self.hparams.use_coco_eval,
                                        use_label_prefix=self.hparams.use_label_prefix,
                                        supervision_tgt=self.hparams.supervision_tgt)
        prompt_tokenizer = train_dataset.tokenizer
        prompt_collator = ClipCocoCollator(tokenizer=prompt_tokenizer)
        self.prompt_collator = prompt_collator
        '''
        train_dataloader = DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                      num_workers=self.hparams.num_workers,
                                      shuffle=True, drop_last=True, collate_fn=prompt_collator)
                                      '''
        log.info(f'Load train set with {len(train_dataset)} examples')

        val_dataset = ClipCocoDataset(model_name=self.hparams.init_model,
                                      split='val', clip_model_type=self.hparams.clip_model_type,
                                      use_caption=self.hparams.use_caption,
                                      label_path=self.hparams.label_path,
                                      sent_init_path=self.hparams.sent_init_path,
                                      eval_label=self.hparams.eval_label,
                                      use_coco_eval=self.hparams.use_coco_eval,
                                      use_label_prefix=self.hparams.use_label_prefix)
        val_dataloader = DataLoader(val_dataset, batch_size=self.hparams.batch_size,
                                    num_workers=self.hparams.num_workers,
                                    shuffle=False, collate_fn=prompt_collator)
        log.info(f'Load val set with {len(val_dataset)} examples')

        if self.hparams.use_open_images:
            open_images_dataset = ClipOpenImagesDataset(model_name=self.hparams.init_model,
                                        split='train', clip_model_type=self.hparams.clip_model_type,
                                        use_caption=self.hparams.use_caption,
                                        label_path=self.hparams.label_path,
                                        sent_init_path=self.hparams.sent_init_path,
                                        fixed_prompt=self.hparams.fixed_prompt,
                                        eval_label=self.hparams.eval_label,
                                        use_coco_eval=self.hparams.use_coco_eval,
                                        use_label_prefix=self.hparams.use_label_prefix)
            log.info(f'Load open images set with {len(open_images_dataset)} examples')
            if self.hparams.use_only_open_images:
                train_dataset = open_images_dataset
            else:
                train_dataset = ConcatDataset(train_dataset, open_images_dataset)
        elif self.hparams.use_goodnews_images:
            goodnews_dataset = ClipGoodNewsDataset(model_name=self.hparams.init_model,
                                        split='train', clip_model_type=self.hparams.clip_model_type,
                                        use_caption=self.hparams.use_caption,
                                        label_path=self.hparams.label_path,
                                        sent_init_path=self.hparams.sent_init_path,
                                        fixed_prompt=self.hparams.fixed_prompt,
                                        eval_label=self.hparams.eval_label,
                                        use_coco_eval=self.hparams.use_coco_eval,
                                        use_label_prefix=self.hparams.use_label_prefix)
            log.info(f'Load goodnews set with {len(goodnews_dataset)} examples')
            train_dataset = ConcatDataset(train_dataset, goodnews_dataset)
        elif self.hparams.debug_replicate_images:
            log.info(f'Replicating train dataset 4 times for debugging')
            train_dataset = ConcatDataset(train_dataset, train_dataset, train_dataset, train_dataset)

        log.info(f'Final training set with {len(train_dataset)} examples')
        train_dataloader = DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                    num_workers=self.hparams.num_workers,
                                    shuffle=True, drop_last=True, collate_fn=prompt_collator)
        self.loaders = {'train': train_dataloader, 'val': val_dataloader}

        log.info(f'Initializing models ...')
        device = self.device
        log.info(f'loading ref policy')
        self.ref_policy = RefPolicy(model_name=self.hparams.ref_model, temperature=self.hparams.temperature,
                                    device=device, model_weight=self.hparams.ref_model_weight)
        log.info(f'loading policy')
        self.policy = Policy(model_name=self.hparams.init_model, temperature=self.hparams.temperature, device=device,
                             clipcap_path=self.hparams.clipcap_path, fix_gpt=self.hparams.fix_gpt,
                             prefix_length=self.hparams.prefix_length,
                             clipcap_num_layers=self.hparams.clipcap_num_layers,
                             label_path=self.hparams.label_path,
                             use_transformer_mapper=self.hparams.use_transformer_mapper,
                             use_ptuning_v2=self.hparams.use_ptuning_v2,
                             model_weight=self.hparams.init_model_weight, use_label_prefix=self.hparams.use_label_prefix)
        if self.hparams.load_finetuned:
            self.policy = load_finetuned(self.hparams, self.policy)
        log.info(f'loading value')
        self.value_model = Value(model_name=self.hparams.init_value_model, device=device,
                                 clipcap_path=self.hparams.clipcap_path, fix_gpt=self.hparams.fix_gpt,
                                 unfix_value_model=self.hparams.unfix_value_model,
                                 prefix_length=self.hparams.prefix_length,
                                 clipcap_num_layers=self.hparams.clipcap_num_layers,
                                 label_path=self.hparams.label_path,
                                 use_transformer_mapper=self.hparams.use_transformer_mapper,
                                 use_ptuning_v2=self.hparams.use_ptuning_v2,
                                 model_weight=self.hparams.init_value_model_weight, use_label_prefix=self.hparams.use_label_prefix)
        rewards = {}
        if self.hparams.use_pair_reward:
            # rewards['pair'] = Reward(gain=None, bias=None,
            rewards['pair'] = Reward(gain=self.hparams.gain, bias=self.hparams.bias,
                                     pair_reward_exp=self.hparams.pair_reward_exp,
                                     clip_model_type=self.hparams.clip_model_type,
                                     clip_sent_split=self.hparams.clip_sent_split, device=device)
        if self.hparams.use_style_reward:
            # rewards['style'] = StyleReward(gain=None, bias=None,
            rewards['style'] = StyleReward(gain=self.hparams.style_gain, bias=self.hparams.style_bias,
                                        style_cache_path=self.hparams.style_cache_path,
                                        label_path=self.hparams.label_path,
                                        style_acc_as_rewards=self.hparams.style_acc_as_rewards,
                                        device=device)
        # normalize the rewards to have mean 0, var 1
        for k, reward in rewards.items():
            reward.set_reward_norm(dataloader=train_dataloader, policy=self.policy)
        self.score_models = nn.ModuleDict(rewards)
        log.info(f'Initialization done!')
        self.labels = getattr(train_dataloader.dataset, 'labels', None)

        self.metrics = Eval()

        if self.hparams.adaptive_kl:
            self.kl_ctl = AdaptiveKLController(self.hparams.kl_coef, params=self.hparams)
        else:
            self.kl_ctl = FixedKLController(self.hparams.kl_coef)

        self.hparams.minibatch_size = exact_div(self.hparams.batch_size, self.hparams.nminibatches)

    def init_buffer(self):
        if self.hparams.use_replay_buffer and not hasattr(self, 'replay'):
            self.replay = Buffer(self.hparams, self.train_dataloader(), self.prompt_collator,
                                 text_model=self.ref_policy,
                                 pair_model=self.score_models['pair'],
                                 device=self.device)

    def train_dataloader(self):
        return self.loaders['train']

    def val_dataloader(self):
        return self.loaders['val']

    def get_kl(self, logprobs, ref_logprobs):
        if self.hparams.kl_type == 'jsd':
            kl = get_jsd(logprobs, ref_logprobs)
        elif self.hparams.kl_type == 'kl_without_sampling':
            kl = get_kl(logprobs, ref_logprobs)
        elif self.hparams.kl_type == 'mean_kl':
            kl = get_mean_kl(logprobs, ref_logprobs)
        else:
            kl = logprobs - ref_logprobs
        return kl

    def get_kl_rewards(self, logprobs, ref_logprobs, masks):
        kl = logprobs - ref_logprobs
        if self.hparams.kl_as_constraints is not None:
            v = self.hparams.kl_as_constraints
            x = masks.sum(-1, keepdim=True)
            x = v / x
            kl = (kl - x).clamp(min=0)
            '''
            sum_kl = reduce_sum(kl, masks, axis=1)
            mask = sum_kl >= v
            mask = mask[:, None]
            kl1 = kl * mask.type_as(kl)
            kl2 = kl * (~mask).type_as(kl)
            kl2 = kl2.clamp(min=0)
            kl = kl1 + kl2
            '''
        if self.hparams.kl_ref_entropy_mask is not None:
            mean_ref_entropy = reduce_sum(-ref_logprobs, masks, axis=1)
            ent_mask = mean_ref_entropy > self.hparams.kl_ref_entropy_mask
            ent_mask = ent_mask[:, None].repeat(1, kl.shape[1])
            kl_clipped = kl.clip(max=-2)
            kl = kl.masked_scatter(ent_mask, kl_clipped)
        reward = -self.kl_ctl.value * kl
        if self.hparams.do_not_use_kl_rewards:
            reward = torch.zeros_like(reward)
        return reward

    def compute_rewards(self, goal_scores, stab_scores, logprobs, ref_logprobs, masks):
        non_score_reward = self.get_kl_rewards(logprobs, ref_logprobs, masks)
        score_reward = 0

        if self.hparams.ref_entropy_mask is not None:
            mean_ref_entropy = reduce_sum(-ref_logprobs, masks, axis=1)
            ent_mask = mean_ref_entropy > self.hparams.ref_entropy_mask
            smask = ent_mask.clone()
            score = torch.tensor(goal_scores).to(smask.device)
            if len(score.shape) == 2:
                smask = smask[:, None].repeat(1, score.shape[1])
            score_clipped = score.clip(max=-1)
            score = score.masked_scatter(smask, score_clipped)
            goal_scores = score.cpu().numpy()

        if not self.use_goal_scores:
            goal_scores = None
        all_scores = [goal_scores, *stab_scores]
        response_length = non_score_reward.shape[1]
        for scores in all_scores:
            if scores is None:
                continue
            scores = np.array(scores)
            if len(scores.shape) == 1:
                reward = torch.tensor([[0.] * (l-1) + [s] + [0.] * (response_length - l) for s, l
                                       in zip(scores, torch.sum(masks, dim=1).tolist())], device=logprobs.device)
            else:  # 2
                reward = torch.tensor([s[:l].tolist() + [0.] * (response_length - l) for s, l
                                       in zip(scores, torch.sum(masks, dim=1).tolist())], device=logprobs.device)
            score_reward += reward
        rewards = non_score_reward + score_reward
        return rewards, goal_scores, non_score_reward, self.kl_ctl.value

    def train_minibatch(self, rollouts):
        """One step of PPO training."""
        opt = self.optimizers()
        opt.zero_grad()
        ppo_loss, stats = self.loss(rollouts)
        self.manual_backward(ppo_loss)
        if self.hparams.clip_grad:
            torch.nn.utils.clip_grad_norm_(itertools.chain(self.policy.model.parameters(),
                                                           self.value_model.model.parameters()),
                                           self.hparams.max_grad_norm)
        opt.step()
        return stats

    def run_train_loop(self, rollouts):
        stat_list = []

        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(self.hparams.noptepochs):
            order = np.random.permutation(self.hparams.batch_size)
            for mb_start in range(0, self.hparams.batch_size, self.hparams.minibatch_size):
                mb_data = {k: v[order[mb_start:mb_start + self.hparams.minibatch_size]] if type(v) == torch.Tensor else
                              [v[i] for i in order[mb_start:mb_start + self.hparams.minibatch_size].tolist()]
                           for k, v in rollouts.items() if v is not None}
                stats = self.train_minibatch(mb_data)
                stat_list.append(stats)

        sch = self.lr_schedulers()
        sch.step()
        # Collect the stats. (They will be averaged later.)
        return {k: [s[k] for s in stat_list] for k in stat_list[0].keys()}

    def training_step(self, batch):
        step_started_at = time.time()

        self.init_buffer()

        assert len(batch['input_ids']) == self.hparams.batch_size, 'insufficient batch'  # this should never happen
        batch_size = len(batch['input_ids'])

        with torch.no_grad():
            pair_scores = None
            rollouts, ref_logprobs, batch, pair_scores = self.get_samples(batch)

            logprobs, masks = rollouts['response/log_prob'], rollouts['response/mask']
            pair_scores = None
            style_scores = None
            entropy_scores = None
            rep_scores = None
            style_acc = None
            acc = None

            responses = rollouts['response/text']
            queries = batch['prefixes']
            texts = [v1 + v2 for v1, v2 in zip(queries, responses)]
            visdial_num = 0
            visdial_false_ratio = None
            if 'pair' in self.score_models:
                if pair_scores is None:
                    if self.hparams.use_binary_visdial:
                        visdial_masks = [text.startswith('dialogue:') for text in batch['raw_prefixes']]
                        false_masks = ['False' in text for text in texts]
                        visdial_masks = torch.tensor(visdial_masks).to(self.device).bool()
                        false_masks = torch.tensor(false_masks).to(self.device).bool()
                        false_masks = false_masks & visdial_masks
                        reward_texts = [text.replace('True', '').replace('False', '').strip() for text in texts]
                        pair_scores = self.score_models['pair'].get_reward(batch['features'],
                                                                            reward_texts,
                                                                            f'step{self.get_global_step()}',
                                                                            device=self.device)
                        pair_scores = torch.tensor(pair_scores).to(self.device)
                        pair_scores = pair_scores.masked_scatter(false_masks,
                                                                -pair_scores.clip(-1, 1))
                        pair_scores = pair_scores.detach().cpu().numpy()
                        visdial_false_num = false_masks.float().sum().item()
                        visdial_num = visdial_masks.float().sum().item()
                        if visdial_num > 0:
                            visdial_false_ratio = visdial_false_num / visdial_num
                    else:
                        pair_scores = self.score_models['pair'].get_reward(batch['features'],
                                                                            texts,
                                                                       f'step{self.get_global_step()}',
                                                                        device=self.device)
            if 'style' in self.score_models:
                style_scores, acc = self.score_models['style'].get_reward(batch['labels'], texts,
                                                                          f'step{self.get_global_step()}',
                                                                        device=self.device)
            if pair_scores is None:
                goal_scores = style_scores
            elif style_scores is None:
                goal_scores = pair_scores
            else:
                goal_scores = pair_scores + style_scores
            stab_scores = []
            if self.hparams.entropy_reward_threshold is not None:
                x = -ref_logprobs.detach()
                m = masks.sum(-1).unsqueeze(-1)
                x = (x - self.hparams.entropy_reward_threshold / m).clamp(min=0)
                entropy_scores = -(x * self.hparams.entropy_reward_coef)
                entropy_scores += self.hparams.entropy_reward_bias / m
                entropy_scores = entropy_scores.cpu().numpy()
                stab_scores.append(entropy_scores)
            if self.hparams.repetition_penalty is not None:
                if self.hparams.use_gpt_tok_for_rp:
                    rep_scores = get_gpt_repetition_penalty(texts, self.policy.tokenizer)
                else:
                    rep_scores = get_whitespace_repetition_penalty(texts)
                rep_scores = rep_scores * self.hparams.repetition_penalty
                rep_scores += self.hparams.repetition_reward_bias
                stab_scores.append(rep_scores)

            rewards, cut_goal_scores, non_score_reward, kl_coef = self.compute_rewards(
                goal_scores, stab_scores, logprobs, ref_logprobs, masks)
            if cut_goal_scores is not None:
                goal_scores = cut_goal_scores
            rollouts['rewards'] = rewards
            rollouts['features'] = batch['features']
            rollouts['prefixes'] = batch['prefixes']
            rollouts['labels'] = batch['labels']
            rollouts['tgt_input_ids'] = batch['tgt_input_ids']
            rollouts['tgt_attention_mask'] = batch['tgt_attention_mask']
            rollouts['tgt_init_input_ids'] = batch['tgt_init_input_ids']
            rollouts['tgt_init_attention_mask'] = batch['tgt_init_attention_mask']

        train_stats = self.run_train_loop(rollouts=rollouts)

        data = {'scores': rewards - non_score_reward, 'total_rewards': rewards,
                'goal_scores': goal_scores,
                'logprobs': logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'non_score_reward': non_score_reward, 'train_stats': train_stats, 'kl_coef': kl_coef,
                'pair_scores': pair_scores, 'style_scores': style_scores, 'style_accuracy': acc,
                'entropy_scores': entropy_scores, 'rep_scores': rep_scores,
                'eos_prob': rollouts['response/eos_prob'], 'ref_eos_prob': rollouts['response/ref_eos_prob']}
        stats = self.get_step_stats(data, self.get_global_step())

        for metric in ['kl', 'kl_coef', 'entropy', 'ref_entropy',
                       'reward_score', 'reward_nonscore', 'reward_total', 'reward_goal',
                       'reward_pair', 'reward_style',
                       'reward_ref_entropy', 'reward_repetition',
                       'clip_cos_sim', 'reward_style_raw',
                       'clip_score', 'style_accuracy',
                       'eos_prob', 'ref_eos_prob']:
            key = f'objective/{metric}'
            if key in stats:
                v = stats[key]
                if torch.is_tensor(v):
                    v = v.item()
                self.log(f'Objective/{metric}', v, on_step=True, batch_size=batch_size)
                print(f'Objective/{metric}', v)
        for metric in ['policy', 'value', 'total', 'opt',
                       'pg_base', 'pg_clipped', 'vf_base', 'vf_clipped',
                       'kl', 'kl_clipped', 'l2', 'clip', 'entropy', 'cs']:
            name = f'ppo/loss/{metric}'
            if name in stats:
                self.log(f'Loss/{metric}', stats[name], on_step=True, batch_size=batch_size)

        for key, metric in {'policy/approxkl': 'approxkl', 'policy/clipfrac': 'pg_clipfrac',
                            'returns/mean': 'returns/mean', 'returns/var': 'returns/var',
                            'val/vpred': 'vpred', 'val/error': 'verror',
                            'advantages/mean': 'advantages/mean', 'advantages/var': 'advantages/var',
                            'val/clipfrac': 'vf_clipfrac'}.items():
            self.log(f'Stats/{metric}', stats[f'ppo/{key}'], on_step=True, batch_size=batch_size)

        self.log(f'Stats/visdial_num', visdial_num, on_step=True, batch_size=batch_size)
        if visdial_false_ratio is not None:
            self.log(f'Stats/visdial_false_ratio', visdial_false_ratio, on_step=True, batch_size=batch_size)
        self.log(f'Stats/use_ref_policy', float(self.use_ref_policy), on_step=True, batch_size=batch_size)
        self.log(f'Stats/use_only_kl', float(self.use_only_kl), on_step=True, batch_size=batch_size)
        self.log(f'Stats/use_goal_scores', float(self.use_goal_scores), on_step=True, batch_size=batch_size)
        self.log(f'Stats/global_step', float(self.get_global_step()), on_step=True, batch_size=batch_size)
        kl_type = {'kl': 0, 'kl_without_sampling': 1, 'jsd' : 2, 'mean_kl': 3}[self.hparams.kl_type]
        self.log(f'Stats/kl_type', kl_type, on_step=True, batch_size=batch_size)

        hypos_log = [f'({q.strip()}) {r.strip()}' for q, r in zip(
            rollouts['query/text'], rollouts['response/text']
        )]

        if self.hparams.use_replay_buffer:
            self.log(f'Replay/offline/size', self.replay.get_size('offline'), on_step=True, batch_size=batch_size)
            self.log(f'Replay/online/size', self.replay.get_size('online'), on_step=True, batch_size=batch_size)
            self.log(f'Replay/offline/age', self.replay.get_ages('offline'), on_step=True, batch_size=batch_size)
            self.log(f'Replay/online/age', self.replay.get_ages('online'), on_step=True, batch_size=batch_size)
            self.log(f'Replay/offline/mean_selected', self.replay.get_num_selecteds('offline'), on_step=True, batch_size=batch_size)
            self.log(f'Replay/online/mean_selected', self.replay.get_num_selecteds('online'), on_step=True, batch_size=batch_size)
            self.log(f'Replay/online/clip_score', self.replay.get_clip_score('online'), on_step=True, batch_size=batch_size)
            self.log(f'Replay/online/rep_score', self.replay.get_rep_score('online'), on_step=True, batch_size=batch_size)
            self.log(f'Replay/online/ref_entropy', self.replay.get_ref_entropy('online'), on_step=True, batch_size=batch_size)
            self.log(f'Replay/gen/clip_score', self.replay.get_clip_score('new_gen'), on_step=True, batch_size=batch_size)
            self.log(f'Replay/gen/rep_score', self.replay.get_rep_score('new_gen'), on_step=True, batch_size=batch_size)
            self.log(f'Replay/gen/ref_entropy', self.replay.get_ref_entropy('new_gen'), on_step=True, batch_size=batch_size)
            replay_names = rollouts['replay_names']
            self.log(f'Replay/offline/sample_ratio', len([v for v in replay_names if v == 'offline']) / len(replay_names),
                     on_step=True, batch_size=batch_size)
            self.log(f'Replay/online/sample_ratio', len([v for v in replay_names if v == 'online']) / len(replay_names),
                     on_step=True, batch_size=batch_size)
            self.log(f'Replay/gen/sample_ratio', len([v for v in replay_names if v == 'new_gen']) / len(replay_names),
                     on_step=True, batch_size=batch_size)
            self.log(f'Replay/gen/kl', self.replay.get_kl('new_gen', self.policy, self.device),
                     on_step=True, batch_size=batch_size)
            offline_ratio = self.replay.get_offline_ratio(self.get_global_step())
            self.log(f'Replay/offline/ratio', offline_ratio, on_step=True, batch_size=batch_size)

        mean_num_rep = np.array([count_rep(v, 2) for v in texts]).mean()
        self.log(f'Stats/rep_score', mean_num_rep, on_step=True, batch_size=batch_size)
        n_words = list(itertools.chain(*[v.split() for v in texts]))
        n_unique_words = set(n_words)
        self.log(f'Stats/doc_unique_words_ratio', len(n_words) / len(n_unique_words), on_step=True, batch_size=batch_size)

        kl = stats['objective/kl']
        kl = torch.Tensor([kl]).to(self.device)
        kl = self.all_gather(kl).mean().item()
        self.update_kl_ctl(kl)

        self.log_train(hypos_log, batch['image_ids'], rollouts,
                       pair_scores, batch['labels'], logprobs, ref_logprobs,
                       masks, rewards, step_started_at)

    def get_samples(self, batch):
        pair_score = None
        with torch.no_grad():
            for tries in range(5):
                if self.use_ref_policy:
                    rollouts = self.ref_policy.sample(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                                      invalidate_eos=not self.hparams.eos_ok,
                                                      no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                                                      max_len=self.hparams.response_length,
                                                      device=self.device)
                else:
                    rollouts = self.policy.sample(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                                  invalidate_eos=not self.hparams.eos_ok,
                                                  no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                                                  features=batch['features'],
                                                  labels=batch['labels'],
                                                  max_len=self.hparams.response_length,
                                                  device=self.device)
                    log_prob = rollouts['response/log_prob']
                    max_entropy = torch.max(reduce_sum(-log_prob, rollouts['response/mask'], axis=1))
                    mean_entropy = torch.mean(reduce_sum(-log_prob, rollouts['response/mask'], axis=1))

                    if (not self.hparams.sampling_entropy_threshold) or \
                            (max_entropy.item() < self.hparams.sampling_entropy_threshold):
                        break
            if self.hparams.use_replay_buffer:
                batch_size = len(batch['image_ids'])
                self.replay.insert(rollouts, batch,
                                   text_model=self.ref_policy,
                                   pair_model=self.score_models['pair'],
                                   buffer_name='online',
                                   device=self.device)
                self.replay.remove(min(self.hparams.max_buffer_size,
                                       max(self.get_global_step() * self.hparams.buffer_growth,
                                           self.hparams.min_buffer_size)
                                       )
                                   )

                offline_ratio = self.replay.get_offline_ratio(self.get_global_step())
                rollouts, batch, pair_score = self.replay.sample(batch_size,
                                                                 offline_ratio,
                                                                 self.device)
            else:
                forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                                'query_mask': rollouts['query/mask'],
                                'response_input_ids': rollouts['response/input_ids'],
                                'response_mask': rollouts['response/mask'],
                                'features': batch['features'],
                                'labels': batch['labels']}

                ref_rollouts = self.ref_policy.forward_pass(**{k: v for k, v in forward_inputs.items()
                                                           if k not in ['features', 'labels']},
                                                            invalidate_eos=not self.hparams.eos_ok,
                                                            device=self.device)
                rollouts = {**rollouts, 'response/ref_log_prob': ref_rollouts['response/log_prob'],
                        'response/ref_eos_prob': ref_rollouts['response/eos_prob']}

            forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                              'query_mask': rollouts['query/mask'],
                              'response_input_ids': rollouts['response/input_ids'],
                              'response_mask': rollouts['response/mask'],
                              'features': batch['features'],
                              'labels': batch['labels']}

            outputs = self.policy.forward_pass(**forward_inputs, invalidate_eos=not self.hparams.eos_ok,
                                                device=self.device)
            rollouts['response/log_prob'] = outputs['response/log_prob']
            rollouts['response/value'] = self.value_model.forward_pass(**forward_inputs,
                                                                       device=self.device)['response/value']
            rollouts['response/value'] *= rollouts['response/mask']

            ref_logprobs = rollouts['response/ref_log_prob']

        return rollouts, ref_logprobs, batch, pair_score

    def update_kl_ctl(self, kl):
        self.kl_ctl.update(kl, self.hparams.batch_size)

    @rank_zero_only
    def log_train(self, hypos_log, image_ids, rollouts, pair_scores, labels, logprobs, ref_logprobs,
                  masks, rewards, step_started_at):

        self.print_samples(image_ids=image_ids, texts=hypos_log, pair_scores=pair_scores,
                           labels=labels,
                           logprobs=logprobs, ref_logprobs=ref_logprobs, masks=masks,
                           rewards=rewards,
                           replay_names=rollouts.get('replay_names', None))

        '''
        step_time = time.time() - step_started_at
        eps_per_second = float(self.hparams.batch_size) / step_time
        log.info(f"[ppo_step {self.global_step}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")
        '''

    def get_step_stats(self, data, step):
        masks = data['masks']
        kl = data['logprobs'] - data['ref_logprobs']
        mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        mean_ref_entropy = torch.mean(reduce_sum(-data['ref_logprobs'], masks, axis=1))
        mean_non_score_reward = torch.mean(reduce_sum(data['non_score_reward'], masks, axis=1))
        stats = {
            'objective/kl': mean_kl.item(),
            'objective/kl_coef': data['kl_coef'],
            'objective/entropy': mean_entropy.item(),
            'objective/ref_entropy': mean_ref_entropy.item(),
        }
        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = np.mean([x.item() if torch.is_tensor(x) else x for x in v])
        stats['objective/eos_prob'] = torch.mean(reduce_mean(data['eos_prob'], masks, axis=1))
        stats['objective/ref_eos_prob'] = torch.mean(reduce_mean(data['ref_eos_prob'], masks, axis=1))
        score_rewards = torch.mean(reduce_sum(torch.tensor(data['scores']).to(masks.device), masks, axis=1))
        total_rewards = torch.mean(reduce_sum(torch.tensor(data['total_rewards']).to(masks.device), masks, axis=1))
        stats['objective/reward_goal'] = np.mean(data['goal_scores'])
        stats['objective/reward_score'] = score_rewards.item()
        stats['objective/reward_nonscore'] = mean_non_score_reward.item()
        stats['objective/reward_total'] = total_rewards.item()
        if 'pair_scores' in data and data['pair_scores'] is not None:
            stats['objective/reward_pair'] = np.mean(data['pair_scores'])
            cos_sim = self.score_models['pair'].unnormalize(data['pair_scores'])
            stats['objective/clip_cos_sim'] = np.mean(cos_sim)
            stats['objective/clip_score'] = np.mean(np.maximum(cos_sim, 0) * 2.5)
        if 'style_scores' in data and data['style_scores'] is not None:
            stats['objective/reward_style'] = np.mean(data['style_scores'])
            stats['objective/reward_style_raw'] = np.mean(self.score_models['style'].unnormalize(data['style_scores']))
        if 'entropy_scores' in data and data['entropy_scores'] is not None:
            scores = torch.tensor(data['entropy_scores']).to(masks.device)
            scores = reduce_sum(scores, masks, axis=1)
            stats['objective/reward_ref_entropy'] = torch.mean(scores)
        if 'rep_scores' in data and data['rep_scores'] is not None:
            stats['objective/reward_repetition'] = np.mean(data['rep_scores'])
        if 'style_accuracy' in data and data['style_accuracy'] is not None:
            stats['objective/style_accuracy'] = np.mean(data['style_scores'])

        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        steps = step + 1
        stats.update({
            'elapsed/updates': steps,
            'elapsed/steps/serial': steps * self.hparams.response_length,
            'elapsed/steps/total': steps * self.hparams.batch_size * self.hparams.response_length,
            'elapsed/episodes': steps * self.hparams.batch_size,
        })
        return stats

    def print_samples(self, image_ids, texts, pair_scores, labels, logprobs, ref_logprobs, masks,
                      rewards, replay_names=None):
        if self.get_global_step() % self.hparams.log_interval != 0:
            return
            # Log samples
        cos_sim = self.score_models['pair'].unnormalize(pair_scores)
        clip_score = np.maximum(cos_sim, 0) * 2.5
        kls = reduce_sum(logprobs - ref_logprobs, masks, axis=1)
        ref_ent = reduce_sum(-ref_logprobs, masks, axis=1)
        ent = reduce_sum(-logprobs, masks, axis=1)
        rewards = reduce_sum(rewards, masks, axis=1)

        def log_(i, log_replay=False):
            sample_kl = kls[i].item()
            print(f"------")
            print(f"|  image_id : {image_ids[i]}")
            if log_replay:
                print(f"|  replay_name : {replay_names[i]}")
            label = None
            if labels is not None and self.labels is not None:
                label = self.labels[labels[i].item()]
                print(f"|  label : {label}")
            print(f"|  text : {texts[i]}")
            print(f"|  score : {clip_score[i]:+.2f}")
            print(f"|  kl : {sample_kl:+.2f}")
            print(f"|  ref_ent : {ref_ent[i]:+.2f}")
            print(f"|  ent : {ent[i]:+.2f}")
            print(f"|  total : {rewards[i]:+.2f}")
            # print(f"|  total : {pair_scores[i] - self.kl_ctl.value * sample_kl:+.2f}")
            print(f"------")

            report = {'id': image_ids[i], 'label': label, 'text': texts[i],
                      'score': f"{clip_score[i]:+.2f}",
                      'kl': f"{sample_kl:+.2f}"}
            if log_replay:
                report['replay_name'] = replay_names[i]
            report = json.dumps(report, indent=4)
            tensorboard = self.logger.experiment
            tensorboard.add_text(f'train/samples/{i}', report, self.get_global_step())

        if replay_names is None:
            for i in range(min(5, len(texts))):
                log_(i, False)
        else:
            x = np.array(replay_names)
            z = np.unique(x)
            match = x[None, :] == z[:, None]
            show_ids = []
            for m in match:
                ids = m.nonzero()[0][:2].tolist()
                show_ids.extend(ids)
            for i in show_ids:
                log_(i, True)

    def validation_step(self, batch, batch_idx):
        if self.hparams.use_coco_eval:
            self.eval_coco(batch, batch_idx)
        else:
            self.eval_base(batch, batch_idx)

    def eval_base(self, batch, batch_idx):
        perplexities, stats = [], defaultdict(lambda: [])
        batch_size = len(batch['image_ids'])
        with torch.no_grad():
            rollouts = self.policy.sample(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                            features=batch['features'], labels=batch['labels'],
                                            max_len=self.hparams.response_length,
                                          device=self.device)
            forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                                'query_mask': rollouts['query/mask'],
                                'response_input_ids': rollouts['response/input_ids'],
                                'response_mask': rollouts['response/mask'],
                                'features': batch['features'],
                                'labels': batch['labels']}
            ref_logprobs = self.ref_policy.forward_pass(**{k: v for k, v in forward_inputs.items()
                                                            if k not in ['features', 'labels']},
                                                        invalidate_eos=not self.hparams.eos_ok,
                                                        device=self.device)['response/log_prob']
            perplexity = -1. * reduce_sum(ref_logprobs, rollouts['response/mask'].type_as(ref_logprobs), axis=1)
            perplexities.extend(perplexity.cpu().detach().numpy().tolist())

            pair_scores = None
            style_scores = None
            responses = rollouts['response/text']
            queries = batch['prefixes']
            texts = [v1 + v2 for v1, v2 in zip(queries, responses)]
            if 'pair' in self.score_models:
                pair_scores = self.score_models['pair']._get_reward(batch['features'],
                                                                    texts,
                                                                    device=self.device)
            if 'style' in self.score_models:
                style_scores, acc = self.score_models['style']._get_reward(batch['labels'], texts,
                                                                           device=self.device)
            if pair_scores is not None:
                stats['pair'].extend(pair_scores)
            if style_scores is not None:
                stats['style'].extend(style_scores)
                stats['style_acc'].extend(acc)

        ppl_score = np.mean(perplexities)
        print(f"  perplexity = {ppl_score:+.2f}")
        self.log('Evaluation/perplexity', ppl_score, on_epoch=True, batch_size=batch_size)

        for k, v in stats.items():
            v_score = np.mean(v)
            print(f"  {k} = {v_score:+.2f}")
            self.log(f'Evaluation/{k}', v_score, on_epoch=True, batch_size=batch_size)

    def eval_coco(self, batch, batch_idx):
        data_hypos, data_tgts, data_image_ids = [], [], []
        data_hypos_log = []
        perplexities, stats = [], defaultdict(lambda: [])
        batch_size = len(batch['image_ids'])
        with torch.no_grad():
            rollouts = self.policy.sample(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                          features=batch['features'], labels=batch['labels'],
                                          sample=False,
                                            max_len=self.hparams.response_length,
                                            device=self.device)
            forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                                'query_mask': rollouts['query/mask'],
                                'response_input_ids': rollouts['response/input_ids'],
                                'response_mask': rollouts['response/mask'],
                                'features': batch['features'],
                                'labels': batch['labels']}
            ref_logprobs = self.ref_policy.forward_pass(**{k: v for k, v in forward_inputs.items()
                                                            if k not in ['features', 'labels']},
                                                        invalidate_eos=not self.hparams.eos_ok,
                                                        device=self.device)['response/log_prob']
            perplexity = -1. * reduce_sum(ref_logprobs, rollouts['response/mask'].type_as(ref_logprobs), axis=1)
            perplexities.extend(perplexity.cpu().detach().numpy().tolist())

            pair_scores = None
            style_scores = None
            responses = rollouts['response/text']
            queries = batch['prefixes']
            starts = batch['raw_prefixes']
            texts = [v1 + v2 for v1, v2 in zip(queries, responses)]
            if 'pair' in self.score_models:
                pair_scores = self.score_models['pair']._get_reward(batch['features'],
                                                                    texts,
                                                                    device=self.device)
            if 'style' in self.score_models:
                style_scores, acc = self.score_models['style']._get_reward(batch['labels'], texts,
                                                                            device=self.device)
            if pair_scores is not None:
                stats['pair'].extend(pair_scores)
            if style_scores is not None:
                stats['style'].extend(style_scores)
                stats['style_acc'].extend(acc)

            hypos = [f'{r.strip()}' for q, r in zip(
                rollouts['query/text'], rollouts['response/text']
            )]
            hypos = [get_first_sentence(v) for v in hypos]
            hypos_log = [f'{q.strip()} {r.strip()}' for q, r in zip(
                rollouts['query/text'], rollouts['response/text']
            )]
            tgts = batch['coco_captions']
            data_hypos.extend(hypos)
            data_hypos_log.extend(hypos_log)
            data_tgts.extend(tgts)
            data_image_ids.extend(batch['image_ids'])

        eval_results = defaultdict(lambda: defaultdict(lambda: []))
        log_results = defaultdict(lambda: defaultdict(lambda: []))
        for image_id, hypo, hypo_raw, tgt in zip(data_image_ids, data_hypos, data_hypos_log, data_tgts):
            eval_results[image_id]['hypo'].append(hypo)
            eval_results[image_id]['tgt'] = tgt
            log_results[image_id]['hypo'].append(hypo)
            log_results[image_id]['tgt'] = tgt
            log_results[image_id]['hypo_raw'].append(hypo_raw)
        eval_results = [{'image_id': k, **v} for k, v in eval_results.items()]
        log_results = [{'image_id': k, **v} for k, v in log_results.items()]
        data_hypos = [get_longest(v['hypo']) for v in eval_results]
        data_tgts = [v['tgt'] for v in eval_results]

        # metrics = self.metrics(data_hypos, data_tgts)

        ppl_score = np.mean(perplexities)
        print(f"  perplexity = {ppl_score:+.2f}")
        self.log('Evaluation/perplexity', ppl_score, on_epoch=True, batch_size=batch_size)

        for k, v in stats.items():
            v_score = np.mean(v)
            print(f"  {k} = {v_score:+.2f}")
            self.log(f'Evaluation/{k}', v_score, on_epoch=True, batch_size=batch_size)
        '''
        for k, v in metrics.items():
            print(f"  {k} = {v:+.2f}")
            self.log(f'Evaluation/{k}', v, on_epoch=True, batch_size=batch_size)
        '''

        self.text_data.extend(log_results)

        if batch_idx < 3:
            print(f'---eval samples {batch_idx}---')
            print(f'hypo: {data_hypos[0]}')
            print(f'tgt: {data_tgts[0][0]}')

    def loss(self, rollouts):
        values = rollouts['response/value']
        old_logprob = rollouts['response/log_prob']
        rewards = rollouts['rewards']
        masks = rollouts['response/mask']

        with torch.no_grad():
            rewards_ = rewards
            if self.hparams.whiten_rewards:
                rewards_ = whiten(rewards, masks, shift_mean=False)

            lastgaelam = 0
            advantages_reversed = []
            # gen_length = self.hparams.response_length
            # gen_length = rollouts['response/input_ids'].shape[1]
            gen_length = masks.shape[1]
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards_[:, t] + self.hparams.gamma * nextvalues - values[:, t]
                lastgaelam = delta + self.hparams.gamma * self.hparams.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            returns = advantages + values

            advantages = whiten(advantages, masks).detach()

        if self.hparams.disable_dropout:
            for module in [*self.policy.children(), *self.value_model.children()]:
                module.eval()

        forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                          'query_mask': rollouts['query/mask'],
                          'response_input_ids': rollouts['response/input_ids'],
                          'response_mask': rollouts['response/mask'],
                          'features': rollouts['features'],
                          'labels': rollouts.get('labels', None)}
        torch.cuda.empty_cache()
        outputs = self.policy.forward_pass(**forward_inputs, invalidate_eos=not self.hparams.eos_ok,
                                           device=self.device)
        outputs['response/value'] = self.value_model.forward_pass(**forward_inputs,
                                                                  device=self.device)['response/value']
        outputs['response/value'] *= rollouts['response/mask']

        vpred = outputs['response/value']
        if self.hparams.alt_loss:
            vf_loss = torch.square(vpred - returns)
            vf_losses1 = vf_loss
            vf_losses2 = vf_loss  # do not use

            vf_loss = reduce_mean(vf_loss, masks)
            vf_clipfrac = vf_loss.detach()  # do not use
        else:
            vpredclipped = clamp(vpred, values - self.hparams.cliprange_value, values + self.hparams.cliprange_value)
            vf_losses1 = torch.square(vpred - returns)
            vf_losses2 = torch.square(vpredclipped - returns)
            vf_loss = .5 * reduce_mean(torch.max(vf_losses1, vf_losses2), masks)
            vf_clipfrac = reduce_mean(torch.gt(vf_losses2, vf_losses1).type_as(vf_losses1), masks)

        logprob = outputs['response/log_prob']
        ratio = torch.exp(logprob - old_logprob)
        if self.hparams.alt_loss:
            pg_losses1 = -advantages * ratio
            '''
            pg_mask = ((advantages >= 0).type_as(advantages) * 2 * self.hparams.cliprange) + \
                (1 - self.hparams.cliprange)
            '''
            pg_mask = torch.clamp(ratio, min=1.0 - self.hparams.cliprange, max=1.0 + self.hparams.cliprange)
            pg_losses2 = -advantages * pg_mask
            pg_loss = reduce_mean(torch.max(pg_losses1, pg_losses2), masks)
            pg_clipfrac = reduce_mean(torch.gt(pg_losses2, pg_losses1).type_as(pg_losses1), masks)
        else:
            pg_losses1 = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio, min=1.0 - self.hparams.cliprange, max=1.0 + self.hparams.cliprange)
            pg_loss = reduce_mean(torch.max(pg_losses1, pg_losses2), masks)
            pg_clipfrac = reduce_mean(torch.gt(pg_losses2, pg_losses1).type_as(pg_losses1), masks)

        pg_losses1 = reduce_mean(pg_losses1.detach(), masks)
        pg_losses2 = reduce_mean(pg_losses2.detach(), masks)
        vf_losses1 = reduce_mean(vf_losses1.detach(), masks)
        vf_losses2 = reduce_mean(vf_losses2.detach(), masks)

        entropy = reduce_mean(outputs['response/entropy'], masks)
        approxkl = .5 * reduce_mean(torch.square(logprob.detach() - old_logprob.detach()), masks)

        kl_loss_log = 0
        kl_loss_log_clipped = 0
        ref_logprob = rollouts['response/ref_log_prob'].detach()
        l2_loss_log = 0
        kl_loss = 0
        if self.hparams.supervise_l2:
            encoder_loss = self.policy.model.get_encoder_loss(rollouts['response/input_ids'], rollouts['features'])
            l2_loss = encoder_loss * self.hparams.supervise_kl_coef
            l2_loss_log = l2_loss.item()
        if self.hparams.supervise_ce:
            kl_loss = -logprob * self.hparams.supervise_kl_coef

            kl_loss_log = kl_loss.detach()
            kl_loss_log = torch.mean(reduce_mean(kl_loss_log, masks, axis=1))
            kl_loss_log = kl_loss_log.item()

            kl_loss = torch.mean(reduce_mean(kl_loss, masks, axis=1))
            kl_loss_log_clipped = kl_loss.item()
        elif self.hparams.supervise_kl:
            coef = self.hparams.supervise_kl
            kl_loss = self.get_kl(logprob, ref_logprob)

            kl_loss_log = kl_loss.detach() * self.hparams.supervise_kl_coef
            kl_loss_log = torch.mean(reduce_sum(kl_loss_log, masks, axis=1))
            kl_loss_log = kl_loss_log.item()

            kl_loss = (kl_loss - coef).clamp(min=0) * self.hparams.supervise_kl_coef
            kl_loss = torch.mean(reduce_sum(kl_loss, masks, axis=1))
            kl_loss_log_clipped = kl_loss.item()

        clip_loss = None
        clip_loss_log = 0
        if self.hparams.use_clip_loss:
            clip_loss = get_clip_loss(self.score_models['pair'],
                                      self.policy.tokenizer,
                                      rollouts['features'],
                                      rollouts['prefixes'],
                                      rollouts['response/input_ids'],
                                      outputs['response/logits'],
                                      masks, self.device,
                                      rand_pos=self.hparams.clip_loss_rand_pos,
                                      num_samples=self.hparams.clip_loss_num_samples)
            clip_loss = self.hparams.clip_coef * clip_loss
            clip_loss_log = clip_loss.item()

        entropy_loss = None
        entropy_loss_log = 0
        if self.hparams.use_entropy_loss:
            logprob = outputs['response/log_prob']
            mean_entropy = reduce_sum(-logprob, masks, axis=1)
            entropy_loss = (mean_entropy - self.hparams.entropy_threshold).clamp(min=0)
            '''
            m = masks.sum(-1).unsqueeze(-1)
            entropy_loss = (-logprob - self.hparams.entropy_threshold / m).clamp(min=0)
            mean_entropy = reduce_sum(-logprob, masks, axis=1)
            '''
            entropy_loss = entropy_loss.mean() * self.hparams.entropy_coef
            entropy_loss_log = entropy_loss.item()

        if self.use_only_kl:
            if self.hparams.supervise_l2:
                loss = self.hparams.vf_coef * vf_loss + l2_loss  # do not optimize pg
            else:
                loss = self.hparams.vf_coef * vf_loss + kl_loss  # do not optimize pg
            loss_total = loss.item() + pg_loss.item()
        else:
            if self.hparams.supervise_ce:
                loss = self.hparams.pg_coef * pg_loss + self.hparams.vf_coef * vf_loss  # do not use ce past n step
                loss_total = loss.item() + kl_loss.item()
            else:
                loss = self.hparams.pg_coef * pg_loss + self.hparams.vf_coef * vf_loss + kl_loss
                loss_total = loss.item()

        if clip_loss is not None:
            loss = loss + clip_loss
            loss_total = loss_total + clip_loss.item()
        if entropy_loss is not None:
            loss = loss + entropy_loss
            loss_total = loss_total + entropy_loss.item()

        return_mean, return_var = reduce_mean(returns, masks), reduce_std(returns, masks)
        value_mean, value_var = reduce_mean(values, masks), reduce_std(values, masks)
        advantage_mean, advantage_var = reduce_mean(advantages, masks), reduce_std(advantages, masks)

        verror = (vpred - returns) ** 2
        vpred_log = reduce_mean(vpred, masks)
        verror_log = reduce_sum(verror, masks)

        stats = dict(
            loss=dict(policy=pg_loss.item(), value=vf_loss.item(), total=loss_total, opt=loss.item(),
                      pg_base=pg_losses1.item(), pg_clipped=pg_losses2.item(),
                      vf_base=vf_losses1.item(), vf_clipped=vf_losses2.item(),
                      l2=l2_loss_log, kl=kl_loss_log, kl_clipped=kl_loss_log_clipped,
                      clip=clip_loss_log, entropy=entropy_loss_log),
            policy=dict(entropy=entropy, approxkl=approxkl.item(), clipfrac=pg_clipfrac),
            returns=dict(mean=return_mean, var=return_var),
            advantages=dict(mean=advantage_mean, var=advantage_var),
            val=dict(vpred=vpred_log, error=verror_log,
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var)
        )
        cs_coef = self.hparams.continual_supervision_coef
        if cs_coef > 0:
            cs_loss = self.get_supervision_loss(rollouts)
            loss = loss + cs_loss * cs_coef
            stats['loss']['cs'] = cs_loss.item()
        return loss, flatten_dict(stats, sep='/')

    def get_supervision_loss(self, rollouts):
        forward_inputs = {'query_input_ids': rollouts['tgt_init_input_ids'],
                          'query_mask': rollouts['tgt_init_attention_mask'],
                          'response_input_ids': rollouts['tgt_input_ids'],
                          'response_mask': rollouts['tgt_attention_mask'],
                          'features': rollouts['features'],
                          'labels': rollouts.get('labels', None)}
        torch.cuda.empty_cache()
        outputs = self.policy.forward_pass(**forward_inputs, invalidate_eos=not self.hparams.eos_ok,
                                           device=self.device)
        log_prob = outputs['response/log_prob']
        loss = -log_prob
        masks = rollouts['tgt_attention_mask']
        loss = torch.mean(reduce_mean(loss, masks, axis=1))
        return loss

    def get_optimizer(self, parameters):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            itertools.chain(self.policy.model.parameters(), self.value_model.model.parameters()),
            lr=self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=self.hparams.total_steps)
        scheduler_config = {
            "scheduler": scheduler,
            # "interval": "step",  # manual opt
            # "frequency": 1,
            # "strict": False,
            "name": None,
        }
        return optimizer, scheduler_config

    def configure_optimizers(self):
        optimizer, scheduler_config = self.get_optimizer([*self.parameters()])
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


backbone_keys = ['ref_policy', 'score_models']
class PartialCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        state_dict = {}
        for k in checkpoint['state_dict'].keys():
            save_flag = True
            for backbone_key in backbone_keys:
                if k.startswith(backbone_key):
                    save_flag = False
                    break
            if save_flag:
                state_dict[k] = checkpoint['state_dict'][k]
        checkpoint['state_dict'] = state_dict

        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
            "dirpath": self.dirpath,
            "best_k_models": self.best_k_models,
            "kth_best_model_path": self.kth_best_model_path,
            "kth_value": self.kth_value,
            "last_model_path": self.last_model_path,
        }


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log.info(f'Using {args.num_gpus} GPUS')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    '''
    time = datetime.now()
    date_time = time.strftime("%m-%d-%Y_%H:%M:%S")
    '''
    save_name = Path(args.config).stem
    args.save_dir = os.path.join(args.output_dir, save_name)
    (Path(args.save_dir) / 'lightning_logs').mkdir(exist_ok=True, parents=True)
    checkpoint_path = find_last_checkpoint(args)
    if checkpoint_path is not None:
        print(f'loading from {checkpoint_path}')

    args.total_steps = ceil_div(args.total_episodes, args.batch_size)
    log.info(f'total_steps {args.total_steps}')
    if args.init_model != 'gpt2':
        log.info(f'disabling clipcap pretrained weights due to backbone model diff')
        args.clipcap_path = ''
    log.info(f'Write to output directory: {args.save_dir}')

    logger = pl.loggers.TensorBoardLogger(args.save_dir)
    checkpoint_callback = PartialCheckpoint(
        save_last=True,
        save_top_k=2,
        monitor='Evaluation/CIDEr' if args.use_coco_eval else 'Objective/reward_total',
        mode="max",
        every_n_train_steps=args.save_interval
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    fit_args = {}
    tr_args = dict(logger=logger, default_root_dir=args.save_dir, callbacks=[checkpoint_callback,
                                                                             DumpCallback(),
                                                                             lr_monitor],
                   val_check_interval=args.eval_interval,
                   log_every_n_steps=args.log_interval,
                   gpus=args.num_gpus)
    if args.use_deepspeed and args.num_gpus > 1:
        tr_args = {**tr_args, 'strategy': "deepspeed_stage_2", 'precision': 16}
    if checkpoint_path is not None and not args.disable_auto_resume:
        log.info(f"resuming from previous checkpoint: {checkpoint_path}")
        fit_args = {**fit_args, 'ckpt_path': str(checkpoint_path)}
    if args.fast_dev_run:
        log.info(f"enabling fast_dev_run")
        tr_args = {**tr_args, 'limit_val_batches': 16, 'limit_train_batches': 16}
    if args.grad_acc > 1:
        log.info(f"gradient accumulation: {args.grad_acc}")
        tr_args = {**tr_args, 'accumulate_grad_batches': args.grad_acc}
    download_weights(args)
    '''
    profiler='simple',
    fast_dev_run=True)
    '''
    if args.num_gpus > 1 and not args.use_deepspeed:
        tr_args['strategy'] = DDPStrategy(find_unused_parameters=False) #'ddp'
    model = PPOTrainer(**vars(args))
    trainer = pl.Trainer(**tr_args)
    trainer.strategy.load_model_state_dict = types.MethodType(load_model_state_dict, trainer.strategy)
    trainer.fit(model, **fit_args)


def load_model_state_dict(self, checkpoint) -> None:
    self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)
    # lightning does not restore global step for manual loops. Hence we do it ourselves.
    self.lightning_module.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.optim_step_progress.total.completed \
        = checkpoint['global_step']


if __name__ == "__main__":
    main()
