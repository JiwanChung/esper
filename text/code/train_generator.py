import os
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)

from args import get_args
from loader import get_loaders
from loss import calc_kl_loss
import deepspeed

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Model(pl.LightningModule):
    def __init__(self, transformer, lr=1e-5, num_warmup_steps=1000, max_steps=100000,
                 optimizer='AdamW', max_length=20, use_kl=False, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(transformer, pad_token='<|endoftext|>')
        self.model = AutoModelForCausalLM.from_pretrained(transformer)
        if use_kl:
            self.model_base = AutoModelForCausalLM.from_pretrained(transformer)
            for param in self.model_base.parameters():
                param.requires_grad_(False)

    def forward(self, batch):
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                             labels=batch['input_ids'])
        loss = outputs.loss
        return loss, outputs.logits

    def training_step(self, batch):
        ce_loss, logits = self.forward(batch)
        B = len(batch['labels'])
        self.log(f'train/ce_loss', ce_loss.item(), batch_size=B, on_step=True)
        loss = ce_loss
        if self.hparams.use_kl:
            with torch.no_grad():
                outputs = self.model_base(**{k: v for k, v in batch.items() if k not in ['labels', 'text_labels']})
                base_logits = outputs.logits
            kl_loss = calc_kl_loss(logits, base_logits, batch['attention_mask'])
            loss += kl_loss
            self.log(f'train/kl_loss', kl_loss.item(), batch_size=B, on_step=True)
        self.log(f'train/loss', loss.item(), batch_size=B, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        res = []
        labels = []
        for i, x in enumerate(batch['text_labels']):
            label = x
            if not self.hparams.prefix_label:
                x = 'Q'
            x = self.tokenizer.encode(f'{x}:', return_tensors='pt').to(self.device)
            x = self.model.generate(x, do_sample=True, top_p=0.9,
                                    max_length=self.hparams.max_length,
                                    pad_token_id=self.tokenizer.pad_token_id)
            x = self.tokenizer.decode(x[0])
            res.append(x)
            labels.append(label)

        '''
        B = len(batch['labels'])
        loss = self.forward(batch)
        self.log(f'val/loss', loss.item(), batch_size=B, on_epoch=True, sync_dist=True)
        '''
        tfboard = self.logger.experiment

        print(f'--- eval samples ---')
        texts = []
        for x, y in zip(labels, res):
            text = f'({x}): {y}'
            print(text)
            tfboard.add_text(f'val/text/{x}', text.replace('\n', '  \n'), self.global_step)
            texts.append(text)

    def get_optimizer(self, parameters):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(parameters, lr=self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    self.hparams.num_warmup_steps,
                                                    self.hparams.max_steps)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": False,
            "name": None,
        }
        return optimizer, scheduler_config

    def configure_optimizers(self):
        optimizer, scheduler_config = self.get_optimizer([*self.parameters()])
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


if __name__ == '__main__':
    args = get_args()

    now = datetime.now().strftime("%d-%b-%Y--%H-%M-%S")
    if args.filenames is None:
        fnames = 'all'
    else:
        fnames = ''.join(args.filenames)
    log_dir = f'../data/log/gen/' + f'{args.transformer}/' + fnames
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    transformers.trainer_utils.set_seed(0)
    print('log_dir', log_dir)
    logger = pl.loggers.TensorBoardLogger(log_dir)
    checkpoint_callback = ModelCheckpoint(
        mode='min',
        save_last=True,
        save_top_k=2,
        monitor="train/loss"
    )
    tr_args = dict(logger=logger, default_root_dir=log_dir, callbacks=[checkpoint_callback])
    tr_args['val_check_interval'] = 2000
    loaders = get_loaders(transformer=args.transformer, max_length=args.max_length,
                          filenames=args.filenames,
                          add_eot=args.add_eot,
                          prefix_label=not args.no_prefix_label,
                          batch_size=args.batch_size, is_generator=True)
    num_labels = len(loaders['train'].dataset.labels)
    model = Model(transformer=args.transformer, max_length=args.max_length, use_kl=args.use_kl,
                  prefix_label=not args.no_prefix_label)
    #trainer = pl.Trainer(gpus=1, **tr_args)
    trainer = pl.Trainer(accelerator="gpu", strategy="ddp", **tr_args)
    trainer.fit(model, loaders['train'], loaders['val'])
