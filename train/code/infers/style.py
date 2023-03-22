import pickle
from pathlib import Path

import torch

from .coco import CocoInferDataset


class StyleInferDataset(CocoInferDataset):
    def load_data(self):
        data_dir = Path('../data/coco/cache/clipcap')
        with open(data_dir / f'{self.clip_model_name}_test_vision.pkl', 'rb') as f:
            all_data = pickle.load(f)
        prefixes = all_data["clip_embedding"]  # .reshape(-1, 512)
        captions_raw = all_data["captions"]
        captions_raw = {int(Path(v['filename']).stem.split('_')[-1]): v for v in captions_raw}
        self.prefixes = prefixes
        self.captions_raw = captions_raw

    def get_feature(self, image_id: str):
        dt = self.captions_raw[image_id]
        feature = self.prefixes[dt['clip_embedding']]
        if not torch.is_tensor(feature):
            feature = torch.from_numpy(feature)
        return dt, feature

    def getitem(self, image_id: str, label_text: str = ''):
        res = super().getitem(image_id)
        if not self.fixed_prompt:
            res['caption'] = f"{label_text}:"
        return res
