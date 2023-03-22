import os
import logging
import math
import pickle
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from metric.lang_metrics import Eval
from finetune import Trainer, main
from infers.common import load_model
from data import ClipCocoCollator


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class VisualNewsDataset(Dataset):
    def __init__(self, model_name, clip_model_type: str, split: str = 'train'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='<|endoftext|>')
        self.split = split
        self.clip_model_type = clip_model_type
        self.clip_model_name = self.clip_model_type.replace('/', '_')

        self.load_data()
        self.keys = list(self.captions_raw.keys())

    def __len__(self):
        return len(self.keys)

    def load_data(self):
        data_dir = Path('../data/visualnews/cache/clipcap')
        with open(data_dir / f'{self.clip_model_name}_{self.split}_vision.pkl', 'rb') as f:
            all_data = pickle.load(f)
        prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        captions_raw = {v['filename']: v for v in captions_raw}
        self.prefixes = prefixes
        self.captions_raw = captions_raw

    def __getitem__(self, idx):
        key = self.keys[idx]
        dt = self.captions_raw[key]
        feature = torch.from_numpy(self.prefixes[dt['clip_embedding']])
        sent = dt['raw_caption'].strip()
        sent = f'{sent}<|endoftext|>'
        if dt is None:
            return None
        res = {
            'image_id': dt['image_id'],
            'caption': 'news:',
            'prefix': '',
            'coco_caption': [sent],
            'feature': feature
        }
        return res


class VisualNewsTrainer(Trainer):
    def get_train_dataset(self):
        return VisualNewsDataset(model_name=self.hparams.init_model,
                                        clip_model_type=self.hparams.clip_model_type,
                                        split='train')

    def get_val_dataset(self):
        return VisualNewsDataset(model_name=self.hparams.init_model,
                                        clip_model_type=self.hparams.clip_model_type,
                                        split='test')


if __name__ == "__main__":
    main('Evaluation/CIDEr', VisualNewsTrainer)
