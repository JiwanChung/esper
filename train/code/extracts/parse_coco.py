import os
import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from PIL import Image
import skimage.io as io
import numpy as np
import torch
import clip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


CLIP_DIM = 512


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {**self.data[i], 'num': i}

def get_loader(data, batch_size):
    return torch.utils.data.DataLoader(dataset=Dataset(data), batch_size=batch_size, shuffle=False)


def main(root, clip_model_type: str):
    num_gpus = torch.cuda.device_count()
    print("Let's use", num_gpus, "GPUs!")
    out_dir = root / 'coco/cache/clipcap'
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(root / 'coco/captions/dataset_coco.json', 'r') as f:
        data = json.load(f)
    data = data['images']
    tqdm.write('formatting data')
    res = defaultdict(lambda: [])
    split_size = {
        'test': 5000,
        'val': 5000,
        'train': 113287
    }
    splits = set(split_size.keys())
    for row in tqdm(data, total=len(data)):
        split = row['split']
        if split not in splits:
            split = 'train'
        for sent in row['sentences']:
            out = {'id': sent['sentid'], 'image_id': f"{sent['imgid']}",
                   'caption': sent['raw'].strip(),
                   'filename': row['filename']}
            res[split].append(out)

    for split, v in res.items():
        print(f'size: {split} ({len(v)})')

    image_dir = str(root / 'coco/images')
    for split, _ in split_size.items():
        v = res[split]
        tqdm.write(f'running {split}')
        clip_model_name = clip_model_type.replace('/', '_')
        out_path = out_dir / f'{clip_model_name}_{split}_vision.pkl'
        if not out_path.is_file():
            run_split(clip_model_type, split, image_dir, out_path, num_gpus, v, split_names=True)

normalizers = torch.tensor(((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))

class ImageLoader:
    def __init__(self, image_dir, split_names, clip_model_type):
        self.image_dir = Path(image_dir)
        self.split_names = split_names

        self.res = {
            'ViT-B/32': 224,
            'RN50x4': 448
        }[clip_model_type]
        self.preprocess = Compose([
            Resize(self.res, interpolation=BICUBIC),
            CenterCrop(self.res),
            _convert_image_to_rgb,
            # ToTensor(),
            # Normalize(),
        ])

    def __call__(self, x):
        i, fname = x
        try:
            x = self.load_image(fname)
            return i, x
        except Exception as e:
            print(f'Error on loading {fname}: {e}')
            return None

    def load_image(self, fname):
        splitted = fname.split('_')
        if self.split_names and len(splitted) >= 2:
            image_split = splitted[1]
            image_path = self.image_dir / image_split / fname
        else:
            image_path = self.image_dir / fname
        image = io.imread(str(image_path))
        image = Image.fromarray(image)
        image = self.preprocess(image)
        return image


def run_split(clip_model_type: str, split: str, image_dir, out_path, num_gpus, data, split_names=True):
    device = torch.device('cuda:0')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    image_loader = ImageLoader(image_dir, split_names, clip_model_type)
    '''
    model = torch.nn.DataParallel(clip_model)
    model = model.to(device)
    '''
    tqdm.write("%0d captions loaded from json " % len(data))
    all_embeddings = {}
    all_embedding_maps = {}
    all_captions = []
    image_dir = Path(image_dir)
    loader = get_loader(data, batch_size=512)

    with Pool(32) as p:
        for batch in tqdm(loader, total=len(loader)):
            loaded = []
            images = []
            images = list(p.map(image_loader, list(enumerate(batch['filename']))))
            images = [v for v in images if v is not None]
            if len(images) == 0:
                continue
            loaded, images = zip(*images)
            loaded = list(loaded)
            images = list(images)
            images = np.stack([np.asarray(image) for image in images], axis=0)
            images = torch.from_numpy(images).float()
            loaded_set = set(loaded)
            batch = {k: v[loaded] if torch.is_tensor(v)
                    else [v2 for i, v2 in enumerate(v) if i in loaded_set]
                    for k, v in batch.items()}
            with torch.no_grad():
                images = images / 255  # -> [0.0, 1.0]
                mean = normalizers[0].to(device)[None, None, None, :]
                std = normalizers[1].to(device)[None, None, None, :]
                images = (images.to(device) - mean) / std
                images = images.permute(0, 3, 1, 2)
                prefixes = clip_model.encode_image(images).cpu()

            del images

            shape = prefixes.shape[-1]
            assert shape == CLIP_DIM, f'invalid clip dim: {shape}'

            seq = [{k: batch[k][i] for k in batch.keys()} for i in range(len(batch['image_id']))]
            for img_id, prefix, i, row in zip(batch['image_id'], prefixes, batch['num'], seq):
                i = i.item()
                row["clip_embedding"] = i
                all_embeddings[i] = prefix.numpy()
                all_captions.append(row)

            if len(all_captions) % 10000 == 0:
                with open(out_path, 'wb') as f:
                    pickle.dump({"clip_embedding": all_embeddings, "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": all_embeddings, "captions": all_captions}, f)

    tqdm.write("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--data_dir', default="./data", type=str)
    args = parser.parse_args()
    root = Path(args.data_dir)
    exit(main(root, args.clip_model_type))
