import os
import math
import traceback
import json
import logging
import time
import random
import pickle
import argparse
import itertools
import types
from itertools import chain
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
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.upgrade_checkpoint import KEYS_MAPPING as DEPRECATED_CHECKPOINT_KEYS


from arguments import get_args
from data import ClipCocoDataset, ClipCocoCollator
from metric.lang_metrics import Eval
from utils.utils import (
    ensure_dir, ceil_div, exact_div,
    whiten, reduce_mean, reduce_sum, reduce_std,
    clamp, flatten_dict,
    get_first_sentence, remove_eot, get_longest, get_first_dot,
    get_jsd, get_kl, get_mean_kl
)
from infers.common import load_model_args, load_model
from load import find_last_checkpoint, download_weights
from resized_tokenizer import get_tokenizer

from deepspeed.utils.logging import logger, get_current_level
# ignore deprecation warning from deepspeed
# pytorch lightning needs it as of now
logger.setLevel(logging.ERROR)
logger.handlers[0].setLevel(logging.ERROR)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class CocoFinetuneDataset(Dataset):
    def __init__(self, model_name, clip_model_type: str, finetune_sample_text: bool = False,
                 supervision_tgt = None, use_resized_tokenizer: bool = False):
        if use_resized_tokenizer:
            self.tokenizer = get_tokenizer(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='<|endoftext|>')
        self.clip_model_type = clip_model_type
        self.clip_model_name = self.clip_model_type.replace('/', '_')
        self.finetune_sample_text = finetune_sample_text
        self.ann = self.load_ann()

        if supervision_tgt is not None:
            path = Path(supervision_tgt)
            if path.is_file():
                log.info(f'loading supervision tgt from: {path}')
                with open(path) as f:
                    x = json.load(f)
                self.supervision_tgt = x
            self.ann = dict(self.ann)
            self.ann = list({k: v for k, v in self.ann.items() if k in self.supervision_tgt.keys()}.items())

        self.load_data()

    def __len__(self):
        return len(self.ann)

    def load_data(self):
        data_dir = Path('../data/coco/cache/clipcap')
        for split in ['train']:
            with open(data_dir / f'{self.clip_model_name}_{split}_vision.pkl', 'rb') as f:
                all_data = pickle.load(f)
            prefixes = all_data["clip_embedding"]
            captions_raw = all_data["captions"]
            captions_raw = {v['filename']: v for v in captions_raw}
            self.prefixes = prefixes
            self.captions_raw = captions_raw

    def load_ann(self):
        ann_path = Path('../data/coco/captions/dataset_coco.json')
        with open(ann_path) as f:
            ann = json.load(f)
        ann = ann['images']
        # ann = {'/'.join(v['filepath'], v['filename']): [sent['raw'] for sent in v['sentences']]
        ann = {str(v['imgid']): {'image_id': str(v['imgid']), 'filename': str(v['filename']),
                                 'sents': [sent['raw'] for sent in v['sentences']]}
               for v in ann if v['split'] not in ['test', 'val']}
        if self.finetune_sample_text:
            log.info("sampling dataset text")
            ann = list(ann.items())
        else:
            # ann = list(chain(*[[(k, v2) for v2 in v] for k, v in ann.items()]))
            ann = list(ann.items())
        return ann

    def get_feature(self, image_id: str):
        dt = {}
        key = ''
        for k, captions_raw in self.captions_raw.items():
            if image_id in captions_raw:
                key = k
                dt = captions_raw[image_id]
                break
        assert dt, f"image_id ({image_id}) not found"
        return dt, self.prefixes[key][dt['clip_embedding']]

    def __getitem__(self, idx):
        key, dt = self.ann[idx]
        image_id = dt['image_id']
        sent = dt['sents']
        key = dt['filename']
        if hasattr(self, 'supervision_tgt'):
            sent = self.supervision_tgt[image_id]
        if not isinstance(sent, str):
            sent = random.choice(sent)
        sent = sent.strip()
        # add dot as eos
        if not sent.endswith('.'):
            sent = f'{sent}.'
        dt = self.captions_raw[key]
        feature = self.prefixes[dt['clip_embedding']]
        if dt is None:
            return None
        res = {
            'image_id': dt['image_id'],
            'caption': 'caption:',
            'prefix': '',
            'coco_caption': [sent],
            'feature': feature
        }
        return res


class DumpCallback(Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.text_data = []

    def on_validation_epoch_end(self, trainer, pl_module):
        data = pl_module.text_data
        hypos = defaultdict(lambda : [])
        tgts = defaultdict(lambda : [])
        for row in data:
            hypos[row['image_id']].append(row['hypo'])
            tgts[row['image_id']].extend(row['tgt'])
        hypos = {k: get_first_dot(remove_eot(get_longest(v))) for k, v in hypos.items()}
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


class Trainer(pl.LightningModule):
    def __init__(self, **params):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False  # optimize in training_step

    def get_global_step(self):
        # global number of calls for train_step. This takes care of multiple updates and multi gpus
        if self.hparams.num_gpus < 2:
            return self.global_step  # this is sufficient for single-gpu case, weirdly though.
        return self.global_step // (self.hparams.nminibatches * self.hparams.num_gpus)

    @property
    def using_deepspeed(self):
        return self.hparams.use_deepspeed and self.hparams.num_gpus > 1

    def get_train_dataset(self):
        return CocoFinetuneDataset(model_name=self.hparams.init_model,
                                   clip_model_type=self.hparams.clip_model_type,
                                   finetune_sample_text=self.hparams.finetune_sample_text,
                                   supervision_tgt=self.hparams.supervision_tgt,
                                   use_resized_tokenizer=self.hparams.use_resized_tokenizer)

    def get_val_dataset(self):
        return ClipCocoDataset(model_name=self.hparams.init_model,
                            split='test', clip_model_type=self.hparams.clip_model_type,
                            use_caption=self.hparams.use_caption,
                            label_path=self.hparams.label_path,
                            sent_init_path=self.hparams.sent_init_path,
                            fixed_prompt=self.hparams.fixed_prompt,
                            eval_label=self.hparams.eval_label,
                            use_coco_eval=self.hparams.use_coco_eval,
                            use_label_prefix=self.hparams.use_label_prefix,
                            finetune=True,
                            use_resized_tokenizer=self.hparams.use_resized_tokenizer)


    def load_data(self):
        log.info(f'Loading data')
        train_dataset = self.get_train_dataset()
        prompt_tokenizer = train_dataset.tokenizer
        prompt_collator = ClipCocoCollator(tokenizer=prompt_tokenizer, finetune=True)
        self.prompt_collator = prompt_collator
        train_dataloader = DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                      num_workers=self.hparams.num_workers,
                                      shuffle=True, drop_last=True, collate_fn=prompt_collator)
        log.info(f'Load train set with {len(train_dataset)} examples')

        test_dataset = self.get_val_dataset()
        test_dataloader = DataLoader(test_dataset, batch_size=self.hparams.batch_size,
                                    num_workers=self.hparams.num_workers,
                                    shuffle=False, collate_fn=prompt_collator)
        log.info(f'Load val set with {len(test_dataset)} examples')
        self.loaders = {'train': train_dataloader, 'test': test_dataloader}

    def setup(self, stage: str = None):
        device = self.device
        log.info(f'Loading model')
        self.model = load_model(self.hparams, device, finetune=True)
        if self.hparams.use_resized_tokenizer:
            rtokenizer = get_tokenizer(self.hparams.init_model)
            vsize = len(rtokenizer)
            if hasattr(self.model.model, 'gpt'):
                self.model.model.gpt.resize_token_embeddings(vsize)
            else:
                self.model.model.resize_token_embeddings(vsize)

        if hasattr(self.model.model, 'gpt'):
            gpt = self.model.model.gpt
        else:
            gpt = self.model.model
        if self.hparams.fix_gpt:
            log.info(f'fixing gpt weights')
            for param in gpt.parameters():
                param.requires_grad_(False)
        else:
            log.info(f'unfreezing gpt weights')
            for param in gpt.parameters():
                param.requires_grad_(True)

        self.hparams.fixed_prompt = ''
        self.hparams.use_coco_eval = True
        self.hparams.eval_label = 'caption'

        self.load_data()

        if self.hparams.num_epochs is not None:
            log.info(f'Total Epochs {self.hparams.num_epochs}')
            self.hparams.total_steps = math.ceil(self.hparams.num_epochs * len(self.loaders['train']) / self.hparams.grad_acc)
            log.info(f'Total Steps {self.hparams.total_steps}')

        self.metrics = Eval()

        if self.hparams.finetune_do_not_normalize_embedding:
            log.info(f'undoing embedding normalization')
            self.model.model.clip_project.divider = 1

    def train_dataloader(self):
        return self.loaders['train']

    def val_dataloader(self):
        return self.loaders['test']

    def run(self, forward_inputs):
        if self.hparams.disable_dropout:
            self.model.eval()
        outputs = self.model.forward_pass(**forward_inputs, invalidate_eos=False,
                                           device=self.device)
        mask = forward_inputs['response_mask']
        logprob = outputs['response/log_prob']
        loss = reduce_mean(-logprob, mask, axis=-1)
        loss = loss.mean()
        # loss = reduce_mean(-logprob, mask)
        batch_size = mask.shape[0]
        return loss, batch_size

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        if batch_idx % (self.hparams.grad_acc) == 0:
            for sched in self.lr_schedulers():
                sched.step()
            for opt in opts:
                opt.zero_grad()
        forward_inputs = {'query_input_ids': batch['input_ids'],
                          'query_mask': batch['attention_mask'],
                          'response_input_ids': batch['gt_input_ids'][:, :self.hparams.response_length],
                          'response_mask': batch['gt_attention_mask'][:, :self.hparams.response_length],
                          'features': batch['features']}
        loss, batch_size = self.run(forward_inputs)
        self.log(f'Train/Loss', loss.item(), on_step=True, batch_size=batch_size)
        self.manual_backward(loss)
        if (batch_idx + 1) % (self.hparams.grad_acc) == 0:
            for opt in opts:
                opt.step()
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            rollouts = self.model.sample(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                          features=batch['features'], labels=batch['labels'],
                                          sample=False,
                                         invalidate_eos=False,
                                            max_len=self.hparams.response_length,
                                            device=self.device)

            forward_inputs = {'query_input_ids': batch['input_ids'],
                              'query_mask': batch['attention_mask'],
                              'response_input_ids': batch['gt_input_ids'][:, :self.hparams.response_length],
                              'response_mask': batch['gt_attention_mask'][:, :self.hparams.response_length],
                              'features': batch['features']}
            loss, batch_size = self.run(forward_inputs)
            self.log(f'Evaluation/Loss', loss.item(), on_epoch=True, batch_size=batch_size)
            responses = rollouts['response/text']
            queries = batch['prefixes']
            texts = [v1 + v2 for v1, v2 in zip(queries, responses)]
            tgts = batch['coco_captions']
            image_ids = batch['image_ids']

        log_results = [{'image_id': a, 'hypo': b, 'tgt': c} for a, b, c
                       in zip(image_ids, texts, tgts)]
        self.text_data.extend(log_results)

        '''
        batch_size = len(texts)
        metrics = self.metrics(texts, tgts)
        for k, v in metrics.items():
            print(f"  {k} = {v:+.2f}")
            self.log(f'Evaluation/{k}', v, on_epoch=True, batch_size=batch_size)
        '''

        if batch_idx < 3:
            print(f'---eval samples {batch_idx}---')
            print(f'image: {image_ids[0]}')
            print(f'hypo: {texts[0]}')
            print(f'tgt: {tgts[0][0]}')

    def get_optimizer(self, parameters_lm, parameters_project):
        lr_project = self.hparams.lr
        if self.hparams.finetune_project_lr is not None:
            lr_project = self.hparams.finetune_project_lr
        optimizer_lm = getattr(torch.optim, self.hparams.optimizer)(
            parameters_lm,
            lr=self.hparams.lr)
        if not self.hparams.constant_lr:
            scheduler_lm = get_linear_schedule_with_warmup(optimizer_lm, num_warmup_steps=self.hparams.warmup_steps,
                                                        num_training_steps=self.hparams.total_steps)
        else:
            scheduler_lm = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_lm,
                                        lr_lambda=lambda epoch: 1.0,
                                        last_epoch=-1, verbose=False)
        optimizer_project = getattr(torch.optim, self.hparams.optimizer)(
            parameters_project,
            lr=lr_project)
        if not self.hparams.constant_lr:
            scheduler_project = get_linear_schedule_with_warmup(optimizer_project, num_warmup_steps=self.hparams.warmup_steps,
                                                        num_training_steps=self.hparams.total_steps)
        else:
            log.info("using constant lr")
            scheduler_project = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_project,
                                        lr_lambda=lambda epoch: 1.0,
                                        last_epoch=-1, verbose=False)

        return [optimizer_lm, optimizer_project], [scheduler_lm, scheduler_project]

    def configure_optimizers(self):
        return self.get_optimizer([*self.model.model.gpt.parameters()], [*self.model.model.clip_project.parameters()])


def main(monitor_stat='Evaluation/CIDEr', trainer_module=Trainer, val_callback=DumpCallback):
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.checkpoint:
        args = load_model_args(args)
    else:
        args.loaded_init_model = None
    log.info(f'Using {args.num_gpus} GPUS')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    save_name = Path(args.config).stem
    args.save_dir = os.path.join(args.output_dir, save_name)
    (Path(args.save_dir) / 'lightning_logs').mkdir(exist_ok=True, parents=True)
    checkpoint_path = find_last_checkpoint(args)
    if checkpoint_path is not None:
        print(f'loading from {checkpoint_path}')

    if args.num_epochs is None:
        args.total_steps = ceil_div(args.total_episodes, args.batch_size * args.grad_acc)
        log.info(f'total_steps {args.total_steps}')
    if args.init_model != 'gpt2':
        log.info(f'disabling clipcap pretrained weights due to backbone model diff')
        args.clipcap_path = ''
    log.info(f'Write to output directory: {args.save_dir}')

    logger = pl.loggers.TensorBoardLogger(args.save_dir)
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=2,
        monitor=monitor_stat,
        mode="max",
        every_n_train_steps=args.save_interval
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    fit_args = {}
    tr_args = dict(logger=logger, default_root_dir=args.save_dir, callbacks=[checkpoint_callback,
                                                                             val_callback(),
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
        # tr_args = {**tr_args, 'accumulate_grad_batches': args.grad_acc}
    download_weights(args)
    '''
    profiler='simple',
    fast_dev_run=True)
    '''
    if args.num_gpus > 1 and not args.use_deepspeed:
        tr_args['strategy'] = 'ddp'
    model = trainer_module(**vars(args))
    trainer = pl.Trainer(**tr_args)
    trainer.strategy.load_model_state_dict = types.MethodType(load_model_state_dict, trainer.strategy)
    trainer.fit(model, **fit_args)


def load_model_state_dict(self, checkpoint) -> None:
    self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)


if __name__ == "__main__":
    main()
