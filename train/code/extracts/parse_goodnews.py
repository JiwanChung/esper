import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .parse_coco import run_split


def main(root, clip_model_type: str):
    num_gpus = torch.cuda.device_count()
    print("Let's use", num_gpus, "GPUs!")
    tqdm.write('loading data')
    out_dir = root / 'goodnews/cache/clipcap'
    out_dir.mkdir(exist_ok=True, parents=True)
    all_data = {}
    with open(root / 'goodnews/data/news_dataset.json', 'r') as f:
        x = json.load(f)
        all_data['train'] = [v for v in x if v['split'] == 'train']

    with open(root / 'goodnews/data/val.json', 'r') as f:
        all_data['val'] = json.load(f)

    with open(root / 'goodnews/data/test.json', 'r') as f:
        all_data['test'] = json.load(f)

    tqdm.write('formatting data')
    res = {}
    for split, data in all_data.items():
        res[split] = []
        for row in tqdm(data, total=len(data), desc='split'):
            image_id = row['imgid']
            caption = row['sentences'][0]['raw']
            raw_caption = row['sentences_full'][0]['raw']
            out = {'id': image_id, 'image_id': image_id,
                    'caption': caption.strip(),
                    'raw_caption': raw_caption.strip(),
                    'filename': row['filename']}
            res[split].append(out)

    for split, v in res.items():
        print(f'size: {split} ({len(v)})')

    image_dir = str(root / 'goodnews/images/resized')
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
