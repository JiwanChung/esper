import pickle
from pathlib import Path

import torch

from .coco import CocoInferDataset


class NocapsInferDataset(CocoInferDataset):

    def load_data(self):
        data_dir = Path('../data/nocaps/cache/clipcap')
        with open(data_dir / f'{self.clip_model_name}_val_vision.pkl', 'rb') as f:
            all_data = pickle.load(f)
        prefixes = all_data["clip_embedding"]
        if torch.is_tensor(prefixes) and len(prefixes.shape) == 1:
            prefixes = prefixes.reshape(-1, 512)
        captions_raw = all_data["captions"]
        captions_raw = {v['image_id']: v for v in captions_raw}
        self.prefixes = prefixes
        self.captions_raw = captions_raw

    def get_feature(self, image_id: str):
        dt = self.captions_raw[image_id]
        feature = self.prefixes[dt['clip_embedding']]
        if not torch.is_tensor(feature):
            feature = torch.from_numpy(feature)
        return dt, feature
