import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

from .parse_coco import run_split


def main(root, clip_model_type: str):
    num_gpus = torch.cuda.device_count()
    print("Let's use", num_gpus, "GPUs!")
    tqdm.write('loading data')
    out_dir = root / 'visualnews/cache/clipcap'
    out_dir.mkdir(exist_ok=True, parents=True)
    all_data = {}
    with open(root / 'visualnews/origin/data.json', 'r') as f:
        x = json.load(f)

    tqdm.write('formatting data')
    sources = defaultdict(lambda: [])
    for row in tqdm(x, total=len(x), desc='splitting sources'):
        source = row['source']
        sources[source].append(row)

    split_sizes = {
        'train': 100000,
        'val': 10000,
        'test': 10000
    }
    splits = defaultdict(lambda: [])
    num_sources = len(sources)
    all_sizes = sum(split_sizes.values())
    for source, data in tqdm(sources.items(), total=len(sources), desc='getting splits'):
        samples = random.sample(data, all_sizes)
        ptr = 0
        for name, size in split_sizes.items():
            split = samples[ptr: ptr+size]
            splits[name].extend(split)
            ptr += size

    for split, v in splits.items():
        print(f'size: {split} ({len(v)})')

    res = {}
    for split, data in splits.items():
        res[split] = []
        for row in tqdm(data, total=len(data), desc='split'):
            filename = row['image_path']
            caption = row['caption']
            out = {'id': row['id'], 'image_id': filename,
                   'source': row['source'],
                   'topic': row['topic'],
                    'caption': caption,
                    'raw_caption': caption,
                    'filename': filename}
            res[split].append(out)

    image_dir = str(root / 'visualnews/origin')
    clip_model_name = clip_model_type.replace('/', '_')
    for split in ['test', 'val', 'train']:
        v = res[split]
        tqdm.write(f'running {split}')
        out_path = out_dir / f'{clip_model_name}_{split}_vision.pkl'
        if not out_path.is_file():
            run_split(clip_model_type, split, image_dir, out_path, num_gpus, v,
                      split_names=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
