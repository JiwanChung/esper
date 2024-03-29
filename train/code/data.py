import os
import math
import logging
import random
import json
import pickle
from itertools import chain
from pathlib import Path
from collections import defaultdict
from typing import Union, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from resized_tokenizer import get_tokenizer


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


def get_unique(li):
    keys = set()
    res = []
    for row in li:
        iid = row['image_id']
        if iid not in keys:
            keys.add(iid)
            res.append(row)
    return res


class ClipCocoCollator:
    def __init__(self, tokenizer, finetune=False, response_length=None):
        self.tokenizer = tokenizer
        self.finetune = finetune
        self.response_length = response_length

    def __call__(self, sequences):
        image_ids = [sequence['image_id'] for sequence in sequences]
        texts = [sequence['caption'] for sequence in sequences]
        prefixes = [sequence['prefix'] for sequence in sequences]

        images = None
        features = None
        if 'feature' in sequences[0]:
            features = [sequence['feature'] for sequence in sequences]
            features = [torch.from_numpy(feature) if not torch.is_tensor(feature) else feature for feature in features]
            features = torch.stack(features, dim=0)
        if 'image' in sequences[0]:
            images = [sequence['image'] for sequence in sequences]

        coco_captions = [sequence['coco_caption'] for sequence in sequences]

        if isinstance(self.response_length, int):
            encodings_dict = self.tokenizer(texts, return_tensors="pt", padding='max_length',
                                            max_length=self.response_length)
        else:
            encodings_dict = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        labels = None
        if len(sequences) > 0 and 'label' in sequences[0]:
            labels = [sequence['label'] for sequence in sequences]
            labels = torch.tensor(labels).long()

        if any([v is None for v in coco_captions]):
            gt_input_ids = None
            gt_attention_mask = None
        else:
            samples = [random.choice(v) for v in coco_captions]
            if isinstance(self.response_length, int):
                encodings_dict = self.tokenizer(samples, return_tensors="pt", padding='max_length',
                                                max_length=self.response_length)
            else:
                encodings_dict = self.tokenizer(samples, return_tensors="pt", padding=True)
            gt_input_ids = encodings_dict['input_ids']
            gt_attention_mask = encodings_dict['attention_mask']

        if 'supervision_tgt' in sequences[0]:
            tgts = [seq['supervision_tgt'] for seq in sequences]
        else:
            tgts = coco_captions
        if any([v is None for v in tgts]):
            tgt_input_ids = None
            tgt_attention_mask = None
        else:
            samples = [random.choice(v) for v in tgts]
            samples = [f' {v.strip()}' for v in samples]
            if isinstance(self.response_length, int):
                encodings_dict = self.tokenizer(samples, return_tensors="pt", padding='max_length',
                                                max_length=self.response_length)
            else:
                encodings_dict = self.tokenizer(samples, return_tensors="pt", padding=True)
            tgt_input_ids = encodings_dict['input_ids']
            tgt_attention_mask = encodings_dict['attention_mask']

        if 'tgt_init' in sequences[0]:
            inits = [seq['tgt_init'] for seq in sequences]
            if isinstance(self.response_length, int):
                encodings_dict = self.tokenizer(inits, return_tensors="pt", padding='max_length',
                                                max_length=self.response_length)
            else:
                encodings_dict = self.tokenizer(inits, return_tensors="pt", padding=True)
            tgt_init_input_ids = encodings_dict['input_ids']
            tgt_init_attention_mask = encodings_dict['attention_mask']
        else:
            tgt_init_input_ids = tgt_input_ids[:, :1]
            tgt_init_attention_mask = tgt_attention_mask[:, :1]
            tgt_input_ids = tgt_input_ids[:, 1:]
            tgt_attention_mask = tgt_attention_mask[:, 1:]

        batch = {
            'image_ids': image_ids,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'features': features,
            'images': images,
            'labels': labels,
            'gt_input_ids': gt_input_ids,
            'gt_attention_mask': gt_attention_mask,
            'tgt_input_ids': tgt_input_ids,
            'tgt_attention_mask': tgt_attention_mask,
            'tgt_init_input_ids': tgt_init_input_ids,
            'tgt_init_attention_mask': tgt_init_attention_mask,
            'coco_captions': coco_captions,
            'raw_prefixes': texts,
            'prefixes': prefixes
        }
        return batch


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = dict(enumerate(datasets))
        self.labels = self.datasets[0].labels
        self.ids_map = list(chain(*[[(i, j) for j in range(len(dataset))]
                                    for i, dataset in self.datasets.items()]))

    def __len__(self):
        return len(self.ids_map)

    def __getitem__(self, idx):
        dataset, idx = self.ids_map[idx]
        return self.datasets[dataset][idx]


class ClipCocoDataset(Dataset):
    def __init__(self, model_name, split='train', prefix_length: int = 10,
                 clip_model_type: str = 'ViT-B/32', use_caption: bool = True,
                 label_path: str = '', sent_init_path: str = '',
                 use_coco_eval: bool = False,
                 fixed_prompt: str = 'Image of a',
                 eval_prompt: str = 'The photo describes a',
                 eval_label: str = 'caption',
                 use_label_prefix: bool = False,
                 supervision_tgt = None,
                 finetune: bool = False,
                 use_resized_tokenizer: bool = False):
        self.clip_model_type = clip_model_type
        if use_resized_tokenizer:
            print("using resized tokenizer")
            self.tokenizer = get_tokenizer(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='<|endoftext|>')
        self.prefix_length = prefix_length
        self.fix_prompt = use_coco_eval and fixed_prompt
        self.fixed_prompt = fixed_prompt
        self.eval_prompt = eval_prompt
        self.finetune = finetune
        self.split = split
        if self.fix_prompt:
            if use_label_prefix:
                log.info(f'({split}) fixing prompt to: ({eval_label}:)')
            else:
                log.info(f'({split}) fixing prompt to: ({self.fixed_prompt})')
        self.use_label_prefix = use_label_prefix
        self.eval_label = eval_label

        self.load_data()

        if self.use_ids_map:
            self.ids_map = defaultdict(lambda: [])
            for i, v in enumerate(self.captions_raw):
                self.ids_map[v['filename']].append(i)
            self.ids_map = dict(self.ids_map)  # finalize
            self.ids_ = list(self.ids_map.keys())

        self.use_caption = use_caption
        if Path(label_path).is_file():
            with open(label_path) as f:
                labels = json.load(f)
            self.labels = {i: v for v, i in labels.items()}
            self.label_candidates = [i for i, v in self.labels.items() if v not in ['generation', 'corpus']]

        sent_init_path = Path(sent_init_path)
        self.sent_inits = {'tokens': ['The'], 'freqs': [1]}
        self.use_corpus_sent_init = False
        if sent_init_path.is_file():
            log.info(f'using predefined sent init file: {sent_init_path}')
            if sent_init_path.suffix == '.txt':
                self.use_corpus_sent_init = True
                with open(sent_init_path) as f:
                    sent_inits = []
                    for line in f:
                        sent_inits.append(line.strip()[:prefix_length * 3])
                self.sent_inits = sent_inits
            else:
                with open(sent_init_path) as f:
                    sent_inits = json.load(f)
                # get most frequent 300
                sent_init_topk = 300
                tokens, freqs = zip(*sent_inits.items())
                tokens, freqs = np.array(tokens), np.array(freqs)
                ids = np.flip(freqs.argsort(-1))[:sent_init_topk]
                tokens, freqs = tokens[ids].tolist(), freqs[ids].tolist()
                tokens.append(self.eval_prompt)
                freqs.append(sum(freqs) / 40)
                self.sent_inits = {'tokens': tokens, 'freqs': freqs}

        if supervision_tgt is not None:
            path = Path(supervision_tgt)
            if path.is_file():
                log.info(f'loading supervision tgt from: {path}')
                with open(path) as f:
                    x = json.load(f)
                self.supervision_tgt = x

    def load_data(self):
        self.use_ids_map = True
        data_dir = Path('../data/coco/cache/clipcap')
        ann_path = Path('../data/coco/captions/dataset_coco.json')
        with open(ann_path) as f:
            ann = json.load(f)
        ann = ann['images']
        self.ann = {str(v['imgid']): [sent['raw'] for sent in v['sentences']] for v in ann}
        # data_dir = Path('../data/coco/cache/')
        clip_model_name = self.clip_model_type.replace('/', '_')
        with open(data_dir / f'{clip_model_name}_{self.split}_vision.pkl', 'rb') as f:
            all_data = pickle.load(f)
        self.prefixes = all_data["clip_embedding"]
        self.captions_raw = all_data["captions"]

    def get_prompt(self):
        if self.fix_prompt:
            caption = self.fixed_prompt
        elif self.use_caption:
            caption = dt['caption']
        elif self.use_corpus_sent_init:
            caption = random.choice(self.sent_inits)
        else:
            idx = np.random.multinomial(1, self.sent_inits['freqs'], size=1).argmax()
            caption = self.sent_inits['tokens'][idx]
        return caption

    def __len__(self) -> int:
        return len(self.ids_map)

    def get_feature(self, dt):
        return self.prefixes[dt['clip_embedding']]

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        dts = self.ids_map[self.ids_[item]]
        dt_idx = random.choice(dts)
        return self.getitem(dt_idx)

    def getitem(self, dt_idx: int) -> Tuple[torch.Tensor, ...]:
        dt = self.captions_raw[dt_idx]

        caption = self.get_prompt()

        coco_caption = None
        if hasattr(self, 'ann'):
            coco_caption = self.ann[dt['image_id']]
        res = {
            'image_id': dt['image_id'],
            'caption': caption,
            'prefix': caption,
            'coco_caption': coco_caption,
            'feature': self.get_feature(dt)
        }

        if hasattr(self, 'supervision_tgt'):
            res['supervision_tgt'] = self.supervision_tgt[dt['image_id']]

        res['tgt_init'] = self.eval_prompt
        if self.finetune:
            res['caption'] = f'caption:'
            res['prefix'] = f''
        elif hasattr(self, 'labels'):
            label = random.choice(self.label_candidates)
            res['label'] = label
            label_text = self.labels[label]

            if self.use_label_prefix:
                res['tgt_init'] = f'caption:'
                if self.fix_prompt:
                    label_text = self.eval_label
                    res['caption'] = f'{label_text.strip()}:'
                    res['prefix'] = ''
                else:
                    caption = res['caption']
                    if label_text == 'dialogue':
                        caption = 'A:'
                    res['caption'] = f'{label_text.strip()}: {caption.strip()}'
                    res['prefix'] = f'{caption.strip()}'

        return res


class ClipOpenImagesDataset(ClipCocoDataset):
    def load_data(self):
        self.use_ids_map = False
        data_dir = Path('../data/open_images/clipfeat')
        pkls = list(data_dir.glob('*.pkl'))
        data = {}
        ## DEBUG: reduce num
        pkls = pkls[:100]
        for pkl in pkls:
            with open(pkl, 'rb') as f:
                shard = pickle.load(f)
                data = {**data, **shard}

        self.captions_raw = [{'image_id': k, 'feature': v} for k, v in data.items()]

    def __len__(self) -> int:
        return len(self.captions_raw)

    def get_feature(self, dt):
        feature = dt['feature']
        return torch.from_numpy(feature)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        return self.getitem(item)


class ClipGoodNewsDataset(ClipCocoDataset):
    def load_data(self):
        self.use_ids_map = False
        data_dir = Path('../data/open_images/clipfeat')
        pkls = list(data_dir.glob('*.pkl'))
        data = {}
        for pkl in pkls:
            with open(pkl, 'rb') as f:
                shard = pickle.load(f)
                data = {**data, **shard}

        self.captions_raw = [{'image_id': k, 'feature': v} for k, v in data.items()]

    def load_data(self):
        self.use_ids_map = True
        data_dir = Path('../data/goodnews/cache/clipcap')
        ann_path = Path('../data/goodnews/data/news_dataset.json')
        with open(ann_path) as f:
            ann = json.load(f)
        self.ann = {str(v['imgid']): [v['sentences_full'][0]['raw']] for v in ann}
        clip_model_name = self.clip_model_type.replace('/', '_')
        with open(data_dir / f'{clip_model_name}_train_vision.pkl', 'rb') as f:
            all_data = pickle.load(f)
        self.prefixes = all_data["clip_embedding"]
        '''
        self.prefixes = [(i, v) for i, v in enumerate(self.prefixes) if math.prod(v.shape) == 512]
        keys, self.prefixes = zip(*self.prefixes)
        self.prefixes = list(self.prefixes)
        keys = set(keys)
        '''
        self.captions_raw = all_data["captions"]
        # self.captions_raw = [row for row in self.captions_raw if row['clip_embedding'] in keys]

    def __len__(self) -> int:
        return len(self.captions_raw)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        return self.getitem(item)

    def get_feature(self, dt):
        return torch.from_numpy(self.prefixes[dt['clip_embedding']])
