import os
import json
import pickle
import math
import re
import random
import argparse
from pathlib import Path
from itertools import chain, product
from collections import defaultdict

from tqdm import tqdm

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
import jax
import jax.numpy as jnp
import clip_jax
from PIL import Image

from utils.utils import get_chunks


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--offline_path', default='../data/coco', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    args = parser.parse_args()
    return args


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def main():
    args = get_args()
    extractor = Extractor(args)

    run(args, extractor)


def load_image(path):
    try:
        return Image.open(str(path)).convert('RGB')
    except Exception as e:
        print(f'Image loading error: {e}')
        return None


def load_images(root, paths):
    res = []
    for path in paths:
        out = load_image(path)
        if out is not None:
            res.append((path, out))
    chunk, image = zip(*res)
    return chunk, image


class Extractor:
    def __init__(self, args):
        self.devices = jax.local_devices()
        print(f"jax devices: {self.devices}")
        print("loading model")
        image_fn, text_fn, jax_params, jax_preprocess = clip_jax.load(args.clip_model_type, "cpu")
        self._preprocess = jax_preprocess
        self.jax_params = jax.device_put_replicated(jax_params, self.devices)
        self.image_fn = jax.pmap(image_fn)
        '''
        self._preprocess_base = Compose([
            Resize(self.res, interpolation=BICUBIC),
            CenterCrop(self.res),
            _convert_image_to_rgb,
            # ToTensor(),
            # Normalize(),
        ])
        '''
        print("loaded model")

    def preprocess(self, path):
        img = self._preprocess(path)
        return img

    def __call__(self, frs):
        feats = jnp.array(self.image_fn(self.jax_params, frs))
        return feats


def run_with_padding(extractor, frs):
    # run with padding
    batch_size = frs.shape[0]
    devices = extractor.devices
    if frs.shape[0] % len(devices) != 0:
        div = math.ceil(frs.shape[0] / len(devices))
        diff = div * len(devices) - frs.shape[0]
        padder = np.repeat(frs[:1], diff, axis=0)
        frs = np.concatenate([frs, padder], axis=0)
    frs = frs.reshape(len(devices), -1, *frs.shape[1:])
    feats = extractor(frs)
    feats = feats.reshape(-1, feats.shape[-1])
    feats = feats[:batch_size]
    feats = np.array(feats)
    return feats


def normalize(v, eps=1e-16):
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)


def expand(x, length=None):
    if length is None:
        length = max([v.shape[0] for v in x])
    ids = np.linspace(0, x.shape[0] - 1, length)
    ids = np.round(ids).astype(int)
    x2 = np.stack([x[i] for i in ids], axis=0)
    return x2


def run_model(extractor, vis):
    vis = [extractor.preprocess(v) for v in vis]
    vis = np.stack(vis, axis=0)

    vis_feats = run_with_padding(extractor, vis)  # N D
    return vis_feats


def run(args, extractor):
    print("loading data")
    root = Path(args.offline_path)
    image_path = root / 'images'
    # for split in ['val', 'test', 'train']:
    for split in ['val', 'test']:
        out_dir = root / 'cache' / 'jax' / split
        out_dir.mkdir(exist_ok=True, parents=True)

        images = list((image_path / f'{split}2014').glob('*.jpg'))

        chunks = list(get_chunks(images, args.batch_size))

        for i, chunk in tqdm(enumerate(chunks), total=len(chunks)):
            path = out_dir / f'chunk_{i+1}_{len(chunks)}.pkl'
            if path.is_file():
                continue
            chunk, images = load_images(image_path, chunk)
            if len(chunk) > 0:
                feats = run_model(extractor, images)
                res = dict(zip(chunk, feats))
                with open(path, 'wb') as f:
                    pickle.dump(res, f)
    print('done')


if __name__ == '__main__':
    main()
