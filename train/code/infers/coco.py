import json
import pickle
from pathlib import Path

import torch
from transformers import AutoTokenizer


class CocoInferDataset:
    fixed_label = 'caption'

    def __init__(self, model_name: str, clip_model_type: str, use_caption: bool = False,
                 label_path: str = '', use_label_prefix: bool = False, fixed_prompt: str = '',
                 infer_dir = None):
        self.clip_model_type = clip_model_type
        self.use_label_prefix = use_label_prefix
        self.fixed_prompt = fixed_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.clip_model_name = self.clip_model_type.replace('/', '_')
        self.prefixes = {}
        self.captions_raw = {}

        self.load_data()

        if Path(label_path).is_file():
            with open(label_path) as f:
                labels = json.load(f)
            self.labels = labels

    def load_data(self):
        data_dir = Path('../data/coco/cache/clipcap')
        for split in ['train', 'val', 'test']:
            with open(data_dir / f'{self.clip_model_name}_{split}_vision.pkl', 'rb') as f:
                all_data = pickle.load(f)
            prefixes = all_data["clip_embedding"]
            captions_raw = all_data["captions"]
            captions_raw = {v['filename']: v for v in captions_raw}
            self.prefixes[split] = prefixes
            self.captions_raw[split] = captions_raw

    def get_feature(self, image_id: str):
        dt = {}
        key = ''
        for k, captions_raw in self.captions_raw.items():
            if image_id in captions_raw:
                key = k
                dt = captions_raw[image_id]
                break
        assert dt, f"image_id ({image_id}) not found"
        feature = self.prefixes[key][dt['clip_embedding']]
        if not torch.is_tensor(feature):
            feature = torch.from_numpy(feature)
        return dt, feature

    def getitem(self, image_id: str, label_text: str = ''):
        dt, feature = self.get_feature(image_id)
        if dt is None:
            return None
        caption = 'The'
        res = {
            'image_id': dt['image_id'],
            'caption': caption,
            'coco_caption': [''],
            'feature': feature
        }
        if label_text:
            label = self.labels.get(label_text, None)
            if label:
                res['label'] = label
        if self.use_label_prefix:
            res['caption'] = f"{self.fixed_label}:"
            if self.fixed_prompt:
                res['caption'] = f"{res['caption']} {self.fixed_prompt}"
        else:
            if self.fixed_prompt:
                res['caption'] = self.fixed_prompt
            if label_text:
                res['caption'] = f"{label_text}:"
        res['prefix'] = res['caption']
        if 'domain' in dt:
            res['domain'] = dt['domain']
        return res
