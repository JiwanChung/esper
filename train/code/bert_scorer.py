import os
import json
import logging
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from load import load_weights


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class BertScorer(nn.Module):
    def __init__(self, cache_path, label_path, device, max_val_loss=100):
        super().__init__()

        self.device = device
        self.max_val_loss = max_val_loss
        self.cache_path = Path(cache_path)
        self.label_path = Path(label_path)
        with open(self.label_path) as f:
            self.labels = json.load(f)
        self.num_labels = len(self.labels)

        hparams = load_weights(self, AutoModelForSequenceClassification,
                               self.cache_path, 'model', None, num_labels=self.num_labels)
        transformer = hparams['transformer']
        mode = hparams.get('mode', 'ce')  # bce
        self.use_bce = mode != 'ce'
        if self.use_bce:
            log.info('use bce for style scoring')
        self.tokenizer = AutoTokenizer.from_pretrained(transformer)

        self.model.to(self.device)
        self.model.eval()

    def forward(self, texts, labels, device = None):
        if device is None:
            device = self.device
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors='pt', truncation=True,
                                    padding=True)
                                    # max_length=self.max_length)
            inputs = inputs.to(device)
            outputs = self.model.bert(**inputs)
            pooled_output = outputs[1]
            logits = self.model.classifier(pooled_output)
            if self.use_bce:
                weight = torch.ones(self.num_labels) * (self.num_labels - 1)
                weight = weight.to(labels.device)
                tgt = F.one_hot(labels, self.num_labels).type_as(weight)
                '''
                loss_fct = nn.BCEWithLogitsLoss(reduction='none',
                                                pos_weight=weight)
                loss = loss_fct(logits, tgt)
                loss = loss.mean(-1)
                '''
                src = logits.detach()
                src = torch.sigmoid(src)
                prob = src.masked_select(tgt.bool())
                src = (src >= 0.5).type_as(logits)
                acc = src.masked_select(tgt.bool())
                rewards = prob.detach().cpu().numpy()
            else:
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                logits = logits.view(-1, logits.shape[-1])
                labels = labels.view(-1)
                loss = loss_fct(logits, labels)
                acc = (logits.argmax(-1) == labels).type_as(logits)
                loss = loss.clamp(-self.max_val_loss, self.max_val_loss)
                rewards = (-loss).detach().cpu().numpy()
        acc = acc.detach().cpu().numpy()
        return rewards, acc
