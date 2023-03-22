import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .parse_coco import run_split


def main(root, clip_model_type: str):
    num_gpus = torch.cuda.device_count()
    print("Let's use", num_gpus, "GPUs!")
    out_dir = root / 'nocaps/cache/clipcap'
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(root / 'nocaps/nocaps_val.json', 'r') as f:
        data = json.load(f)

    images = {v['id']: v for v in data['images']}
    data = data['annotations']
    tqdm.write('formatting data')
    res = []
    for row in tqdm(data, total=len(data)):
        idx = row['image_id']
        image = images[idx]
        out = {'id': f"{row['id']}", 'image_id': f'{idx}',
                'caption': row['caption'].strip(),
               'domain': image['domain'],
                'filename': f'{idx}.jpg'}
        res.append(out)

    print(f'size: ({len(res)})')

    image_dir = str(root / 'nocaps/images')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = out_dir / f'{clip_model_name}_val_vision.pkl'
    if not out_path.is_file():
        run_split(clip_model_type, 'val', image_dir, out_path, num_gpus, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
