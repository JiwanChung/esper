import os
import traceback
import torch
import json
import time
import logging
import random
import argparse
import itertools
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from arguments import get_args
from data import ClipCocoDataset, ClipCocoCollator
from policy import Policy
from ref_policy import RefPolicy
from value import Value
from reward import Reward
from style_reward import StyleReward
from utils.utils import (
    ensure_dir, ceil_div, exact_div,
    whiten, reduce_mean, reduce_sum, reduce_std,
    clamp, flatten_dict,
    get_first_sentence, get_longest
)
from metric.lang_metrics import Eval


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class AdaptiveKLController:
    def __init__(self, init_kl_coef, params):
        self.value = init_kl_coef
        self.params = params

    def update(self, current, n_steps):
        target = self.params.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.params.horizon
        self.value *= mult


class PPOTrainer:
    def __init__(self,
                 params: argparse.Namespace,
                 policy: Policy,
                 ref_policy: RefPolicy,
                 value_model: Value,
                 score_models: Dict[str, Reward],
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: LambdaLR,
                 tensorboard_dir: str,
                 labels: Dict[int, str]):

        self.params = params
        self.policy = policy
        self.ref_policy = ref_policy
        self.value_model = value_model
        self.score_models = score_models
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.labels = labels
        self.train_sampler = iter(self.train_dataloader)
        self.writer = SummaryWriter(tensorboard_dir)
        self.log_dir = Path(tensorboard_dir).parent

        self.writer.add_hparams({k: getattr(self.params, k) for k in
                                 ['use_style_reward', 'response_length',
                                  'use_pair_reward', 'ref_model', 'fix_gpt']}, {})

        if self.params.adaptive_kl:
            self.kl_ctl = AdaptiveKLController(self.params.kl_coef, params=self.params)
        else:
            self.kl_ctl = FixedKLController(self.params.kl_coef)

        self.params.minibatch_size = exact_div(self.params.batch_size, self.params.nminibatches)

        self.metrics = Eval()

    def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
        kl = logprobs - ref_logprobs
        non_score_reward = -self.kl_ctl.value * kl
        score_reward = torch.tensor([[0.] * (l-1) + [s] + [0.] * (self.params.response_length - l) for s, l
                                     in zip(scores, torch.sum(masks, dim=1).tolist())], device=logprobs.device)
        rewards = non_score_reward + score_reward
        return rewards, non_score_reward, self.kl_ctl.value

    def train_minibatch(self, rollouts):
        """One step of PPO training."""
        self.optimizer.zero_grad()
        ppo_loss, stats = self.loss(rollouts)
        ppo_loss.backward()
        if self.params.clip_grad:
            torch.nn.utils.clip_grad_norm_(itertools.chain(self.policy.model.parameters(),
                                                           self.value_model.model.parameters()),
                                           self.params.max_grad_norm)
        self.optimizer.step()
        return stats

    def train(self, rollouts):
        stat_list = []

        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(self.params.noptepochs):
            order = np.random.permutation(self.params.batch_size)
            for mb_start in range(0, self.params.batch_size, self.params.minibatch_size):
                mb_data = {k: v[order[mb_start:mb_start + self.params.minibatch_size]] if type(v) == torch.Tensor else
                              [v[i] for i in order[mb_start:mb_start + self.params.minibatch_size].tolist()]
                           for k, v in rollouts.items() if v is not None}
                stats = self.train_minibatch(mb_data)
                stat_list.append(stats)

        # Collect the stats. (They will be averaged later.)
        return {k: [s[k] for s in stat_list] for k in stat_list[0].keys()}

    def step(self, step_num):
        step_started_at = time.time()

        try:
            image_ids, input_ids, attention_mask, features, labels, _ = next(self.train_sampler)
            assert len(input_ids) == self.params.batch_size, 'insufficient batch'
        except (StopIteration, AssertionError):
            self.train_sampler = iter(self.train_dataloader)
            image_ids, input_ids, attention_mask, features, labels, _ = next(self.train_sampler)

        with torch.no_grad():
            rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask,
                                          features=features, labels=labels,
                                          max_len=self.params.response_length)
            forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                              'query_mask': rollouts['query/mask'],
                              'response_input_ids': rollouts['response/input_ids'],
                              'response_mask': rollouts['response/mask'],
                              'features': features,
                              'labels': labels}
            rollouts['response/value'] = self.value_model.forward_pass(**forward_inputs)['response/value']
            rollouts['response/value'] *= rollouts['response/mask']

            ref_logprobs = self.ref_policy.forward_pass(**{k: v for k, v in forward_inputs.items() if k not in ['features', 'labels']})['response/log_prob']

        logprobs, masks = rollouts['response/log_prob'], rollouts['response/mask']
        pair_scores = None
        style_scores = None
        style_acc = None
        acc = None
        if 'pair' in self.score_models:
            pair_scores = self.score_models['pair'].get_reward(features, rollouts['response/text'], f'step{step_num}')
        if 'style' in self.score_models:
            style_scores, acc = self.score_models['style'].get_reward(labels, rollouts['response/text'], f'step{step_num}')
        if pair_scores is None:
            scores = style_scores
        elif style_scores is None:
            scores = pair_scores
        else:
            scores = pair_scores + style_scores
        rewards, non_score_reward, kl_coef = self.compute_rewards(scores, logprobs, ref_logprobs, masks)
        rollouts['rewards'] = rewards
        rollouts['features'] = features
        rollouts['labels'] = labels

        train_stats = self.train(rollouts=rollouts)

        data = {'scores': scores, 'logprobs': logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'non_score_reward': non_score_reward, 'train_stats': train_stats, 'kl_coef': kl_coef,
                'pair_scores': pair_scores, 'style_scores': style_scores, 'style_accuracy': acc}
        stats = self.record_step_stats(data, step_num)

        for metric in ['kl', 'kl_coef', 'entropy', 'reward', 'reward_total',
                       'non_score_reward',
                       'reward_pair', 'reward_style',
                       'clip_cos_sim', 'reward_style_raw',
                       'clip_score', 'style_accuracy']:
            key = f'objective/{metric}'
            if key in stats:
                self.writer.add_scalar(f'Objective/{metric}', stats[key], step_num)
        for metric in ['policy', 'value', 'total']:
            self.writer.add_scalar(f'Loss/{metric}', stats[f'ppo/loss/{metric}'], step_num)

        self.kl_ctl.update(stats['objective/kl'], self.params.batch_size)

        texts = [f'({a}) {b}' for a, b in zip(rollouts['query/text'], rollouts['response/text'])]
        self.print_samples(image_ids=image_ids, texts=texts, scores=scores,
                           labels=labels,
                           logprobs=logprobs, ref_logprobs=ref_logprobs, masks=masks, step=step_num)

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params.batch_size) / step_time
        log.info(f"[ppo_step {step_num}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")
        self.save(step=step_num)
        if self.params.use_coco_eval:
            self.eval_coco(step=step_num)
        else:
            self.eval(step=step_num)

    def record_step_stats(self, data, step):
        masks = data['masks']
        kl = data['logprobs'] - data['ref_logprobs']
        mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        mean_non_score_reward = torch.mean(reduce_sum(data['non_score_reward'], masks, axis=1))
        stats = {
            'objective/kl': mean_kl.item(),
            'objective/kl_coef': data['kl_coef'],
            'objective/entropy': mean_entropy.item(),
        }
        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = np.mean([x.item() for x in v])
        stats['objective/reward'] = np.mean(data['scores'])
        stats['objective/reward_total'] = np.mean(data['scores']) + mean_non_score_reward.item()
        if 'pair_scores' in data and data['pair_scores'] is not None:
            stats['objective/reward_pair'] = np.mean(data['pair_scores'])
            cos_sim = self.score_models['pair'].unnormalize(data['pair_scores'])
            stats['objective/clip_cos_sim'] = np.mean(cos_sim)
            stats['objective/clip_score'] = np.mean(np.maximum(cos_sim, 0) * 2.5)
        if 'style_scores' in data and data['style_scores'] is not None:
            stats['objective/reward_style'] = np.mean(data['style_scores'])
            stats['objective/reward_style_raw'] = np.mean(self.score_models['style'].unnormalize(data['style_scores']))
        if 'style_accuracy' in data and data['style_accuracy'] is not None:
            stats['objective/style_accuracy'] = np.mean(data['style_accuracy'])

        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        steps = step + 1
        stats.update({
            'elapsed/updates': steps,
            'elapsed/steps/serial': steps * self.params.response_length,
            'elapsed/steps/total': steps * self.params.batch_size * self.params.response_length,
            'elapsed/episodes': steps * self.params.batch_size,
        })
        return stats

    def print_samples(self, image_ids, texts, scores, labels, logprobs, ref_logprobs, masks, step):
        if step % self.params.log_interval != 0:
            return
            # Log samples
        for i in range(min(5, len(texts))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            print(f"  image_id : {image_ids[i]}")
            label = None
            if labels is not None:
                label = self.labels[labels[i].item()]
                print(f"  label = {label}")
            print(texts[i])
            print(f"  score = {scores[i]:+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {scores[i] - self.kl_ctl.value * sample_kl:+.2f}")

            report = {'id': image_ids[i], 'label': label, 'text': texts[i],
                      'score': f"{scores[i]:+.2f}",
                      'kl': f"{sample_kl:+.2f}"}
            report = json.dumps(report, indent=4)
            self.writer.add_text(f'train/samples/{i}', report, step)

    def save(self, step):
        if step % self.params.save_interval != 0:
            return
        torch.save({
            'policy_model': self.policy.model.state_dict(),
            'value_model': self.value_model.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f'{self.params.model_dir}/ckp_{step}.pth')
        log.info(f"[ppo_step {step}] model checkpoint saved")

    def eval(self, step):
        if step % self.params.eval_interval != 0:
            return
        log.info(f"[ppo_step {step}] evaluating ...")

        perplexities, stats = [], defaultdict(lambda: [])
        for i, (image_ids, input_ids, attention_mask, features, labels, coco_captions) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask,
                                              features=features, labels=labels,
                                              max_len=self.params.response_length)
                forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                                  'query_mask': rollouts['query/mask'],
                                  'response_input_ids': rollouts['response/input_ids'],
                                  'response_mask': rollouts['response/mask'],
                                  'features': features,
                                  'labels': labels}
                ref_logprobs = self.ref_policy.forward_pass(**{k: v for k, v in forward_inputs.items() if k not in ['features', 'labels']})['response/log_prob']
                perplexity = -1. * reduce_sum(ref_logprobs, rollouts['response/mask'].type_as(ref_logprobs), axis=1)
                perplexities.extend(perplexity.cpu().detach().numpy().tolist())

                pair_scores = None
                style_scores = None
                if 'pair' in self.score_models:
                    pair_scores = self.score_models['pair']._get_reward(features,
                                                                        rollouts['response/text'])
                if 'style' in self.score_models:
                    style_scores, acc = self.score_models['style']._get_reward(labels,
                                                                          rollouts['response/text'])
                if pair_scores is not None:
                    stats['pair'].extend(pair_scores)
                if style_scores is not None:
                    stats['style'].extend(style_scores)
                    stats['style_acc'].extend(acc)

        ppl_score = np.mean(perplexities)
        print(f"  perplexity = {ppl_score:+.2f}")
        self.writer.add_scalar('Evaluation/perplexity', ppl_score, step)

        for k, v in stats.items():
            v_score = np.mean(v)
            print(f"  {k} = {v_score:+.2f}")
            self.writer.add_scalar(f'Evaluation/{k}', v_score, step)

    def eval_coco(self, step):
        if step % self.params.eval_interval != 0:
            return
        log.info(f"[ppo_step {step}] evaluating ...")

        data_hypos, data_tgts, data_image_ids = [], [], []
        data_hypos_log = []
        perplexities, stats = [], defaultdict(lambda: [])
        for i, (image_ids, input_ids, attention_mask, features, labels, coco_captions) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask,
                                              features=features, labels=labels,
                                              sample=False,
                                              max_len=self.params.response_length)
                forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                                  'query_mask': rollouts['query/mask'],
                                  'response_input_ids': rollouts['response/input_ids'],
                                  'response_mask': rollouts['response/mask'],
                                  'features': features,
                                  'labels': labels}
                ref_logprobs = self.ref_policy.forward_pass(**{k: v for k, v in forward_inputs.items() if k not in ['features', 'labels']})['response/log_prob']
                perplexity = -1. * reduce_sum(ref_logprobs, rollouts['response/mask'].type_as(ref_logprobs), axis=1)
                perplexities.extend(perplexity.cpu().detach().numpy().tolist())

                pair_scores = None
                style_scores = None
                if 'pair' in self.score_models:
                    pair_scores = self.score_models['pair']._get_reward(features,
                                                                        rollouts['response/text'])
                if 'style' in self.score_models:
                    style_scores, acc = self.score_models['style']._get_reward(labels,
                                                                          rollouts['response/text'])
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
                tgts = coco_captions
                data_hypos.extend(hypos)
                data_hypos_log.extend(hypos_log)
                data_tgts.extend(tgts)
                data_image_ids.extend(image_ids)

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

        metrics = self.metrics(data_hypos, data_tgts)

        ppl_score = np.mean(perplexities)
        print(f"  perplexity = {ppl_score:+.2f}")
        self.writer.add_scalar('Evaluation/perplexity', ppl_score, step)

        for k, v in stats.items():
            v_score = np.mean(v)
            print(f"  {k} = {v_score:+.2f}")
            self.writer.add_scalar(f'Evaluation/{k}', v_score, step)
        for k, v in metrics.items():
            print(f"  {k} = {v:+.2f}")
            self.writer.add_scalar(f'Evaluation/{k}', v, step)

        coco_dir = self.log_dir / 'coco_eval'
        coco_dir.mkdir(exist_ok=True)
        with open(coco_dir / f'step_{step}.json', 'w') as f:
            json.dump(log_results, f, indent=4)

        log_len = 3
        skip = 20
        ids = [i * skip for i in range(log_len)]
        sample_hypos = [data_hypos[i] for i in ids]
        sample_tgts = [data_tgts[i] for i in ids]
        for hypo, tgt in zip(sample_hypos, sample_tgts):
            print(f'---eval samples---')
            print(f'hypo: {hypo}')
            print(f'tgt: {tgt[0]}')

    def loss(self, rollouts):
        values = rollouts['response/value']
        old_logprob = rollouts['response/log_prob']
        rewards = rollouts['rewards']
        masks = rollouts['response/mask']

        with torch.no_grad():
            if self.params.whiten_rewards:
                rewards = whiten(rewards, masks, shift_mean=False)

            lastgaelam = 0
            advantages_reversed = []
            gen_length = self.params.response_length
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + self.params.gamma * nextvalues - values[:, t]
                lastgaelam = delta + self.params.gamma * self.params.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            returns = advantages + values

            advantages = whiten(advantages, masks).detach()

        forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                          'query_mask': rollouts['query/mask'],
                          'response_input_ids': rollouts['response/input_ids'],
                          'response_mask': rollouts['response/mask'],
                          'features': rollouts['features'],
                          'labels': rollouts.get('labels', None)}
        torch.cuda.empty_cache()
        outputs = self.policy.forward_pass(**forward_inputs)
        outputs['response/value'] = self.value_model.forward_pass(**forward_inputs)['response/value']
        outputs['response/value'] *= rollouts['response/mask']

        vpred = outputs['response/value']
        vpredclipped = clamp(vpred, values - self.params.cliprange_value, values + self.params.cliprange_value)
        vf_losses1 = torch.square(vpred - returns)
        vf_losses2 = torch.square(vpredclipped - returns)
        vf_loss = .5 * reduce_mean(torch.max(vf_losses1, vf_losses2), masks)
        vf_clipfrac = reduce_mean(torch.gt(vf_losses2, vf_losses1).type_as(vf_losses1), masks)

        logprob = outputs['response/log_prob']
        ratio = torch.exp(logprob - old_logprob)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, min=1.0 - self.params.cliprange, max=1.0 + self.params.cliprange)
        pg_loss = reduce_mean(torch.max(pg_losses, pg_losses2), masks)
        pg_clipfrac = reduce_mean(torch.gt(pg_losses2, pg_losses).type_as(pg_losses), masks)

        loss = pg_loss + self.params.vf_coef * vf_loss

        entropy = reduce_mean(outputs['response/entropy'], masks)
        approxkl = .5 * reduce_mean(torch.square(logprob - old_logprob), masks)

        return_mean, return_var = reduce_mean(returns, masks), reduce_std(returns, masks)
        value_mean, value_var = reduce_mean(values, masks), reduce_std(values, masks)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl, clipfrac=pg_clipfrac),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=reduce_mean(vpred, masks), error=reduce_mean((vpred - returns) ** 2, masks),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var)
        )
        return loss, flatten_dict(stats, sep='/')


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

    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_name = Path(args.config).stem
    args.save_dir = os.path.join(args.output_dir, save_name, 'orig')
    args.reward_dir = os.path.join(args.save_dir, 'reward')
    args.model_dir = os.path.join(args.save_dir, 'model')
    args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
    for d in [args.output_dir, args.save_dir, args.reward_dir, args.model_dir, args.tensorboard_dir]:
        ensure_dir(d)
    if args.init_model != 'gpt2':
        log.info(f'disabling clipcap pretrained weights due to backbone model diff')
        args.clipcap_path = ''
    log.info(f'Write to output directory: {args.save_dir}')

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    log.info(f'Initializing models ...')
    ref_policy = RefPolicy(model_name=args.ref_model, temperature=args.temperature, device=device,
                           model_weight=args.ref_model_weight)
    policy = Policy(model_name=args.init_model, temperature=args.temperature, device=device,
                    clipcap_path=args.clipcap_path, fix_gpt=args.fix_gpt,
                    label_path=args.label_path,
                    use_transformer_mapper=args.use_transformer_mapper,
                    model_weight=args.init_model_weight, use_label_prefix=args.use_label_prefix)
    value = Value(model_name=args.init_model, device=device,
                  clipcap_path=args.clipcap_path, fix_gpt=args.fix_gpt,
                  label_path=args.label_path,
                  use_transformer_mapper=args.use_transformer_mapper,
                  model_weight=args.init_model_weight, use_label_prefix=args.use_label_prefix)
    if args.only_download:
        return
    rewards = {}
    if args.use_pair_reward:
        # rewards['pair'] = Reward(gain=None, bias=None,
        rewards['pair'] = Reward(gain=args.gain, bias=args.bias,
                        clip_model_type=args.clip_model_type,
                        clip_sent_split=args.clip_sent_split, device=device)
    if args.use_style_reward:
        # rewards['style'] = StyleReward(gain=None, bias=None,
        rewards['style'] = StyleReward(gain=args.style_gain, bias=args.style_bias,
                                       style_cache_path=args.style_cache_path,
                                       label_path=args.label_path,
                                       style_acc_as_rewards=args.style_acc_as_rewards,
                                       device=device)
    log.info(f'Initialization done!')

    train_dataset = ClipCocoDataset(model_name=args.init_model,
                                    split='train', clip_model_type=args.clip_model_type,
                                    use_caption=args.use_caption,
                                    label_path=args.label_path,
                                    sent_init_path=args.sent_init_path,
                                    fixed_prompt=args.fixed_prompt,
                                    use_coco_eval=args.use_coco_eval,
                                    use_label_prefix=args.use_label_prefix)
    prompt_tokenizer = train_dataset.tokenizer
    prompt_collator = ClipCocoCollator(tokenizer=prompt_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, collate_fn=prompt_collator)
    log.info(f'Load train set with {len(train_dataset)} examples')

    val_dataset = ClipCocoDataset(model_name=args.init_model,
                                  split='val', clip_model_type=args.clip_model_type,
                                  use_caption=args.use_caption,
                                  label_path=args.label_path,
                                  sent_init_path=args.sent_init_path,
                                  use_coco_eval=args.use_coco_eval,
                                  use_label_prefix=args.use_label_prefix)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=prompt_collator)
    log.info(f'Load val set with {len(val_dataset)} examples')

    labels = getattr(train_dataset, 'labels', None)

    # normalize the rewards to have mean 0, var 1
    for k, reward in rewards.items():
        reward.set_reward_norm(dataloader=train_dataloader, policy=policy)

    # set up optimizer and scheduler
    optimizer = getattr(torch.optim, args.optimizer)(itertools.chain(policy.model.parameters(), value.model.parameters()), lr=args.lr, eps=1e-5)
    args.total_steps = ceil_div(args.total_episodes, args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.total_steps)
    trainer = PPOTrainer(params=args, policy=policy, ref_policy=ref_policy, value_model=value, score_models=rewards,
                         train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                         optimizer=optimizer, scheduler=scheduler, tensorboard_dir=args.tensorboard_dir,
                         labels=labels)

    log.info(f'total_steps {args.total_steps}')
    for step_num in range(args.total_steps):
        trainer.step(step_num)
        '''
        try:
            trainer.step(step_num)
        except RuntimeError as e:
            traceback.print_tb(e.__traceback__)
            torch.cuda.empty_cache()
            continue
        '''


if __name__ == "__main__":
    main()
