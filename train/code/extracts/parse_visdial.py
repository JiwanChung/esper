import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .parse_coco import run_split


def main(root, clip_model_type: str):
    num_gpus = torch.cuda.device_count()
    print("Let's use", num_gpus, "GPUs!")
    out_dir = root / '/visdial/cache/clipcap'
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(root / 'visdial/visdial_1.0_test.json', 'r') as f:
        data = json.load(f)

    data = data['data']

    tqdm.write('formatting data')
    res = []
    for row in tqdm(data['dialogs'], total=len(data['dialogs'])):
        idx = row['image_id']
        for i, turn in enumerate(row['dialog']):
            question = turn['question']
            if 'answer_options' not in turn:
                continue
            cands = turn['answer_options']
            question = data['questions'][question]
            cands = [data['answers'][cand] for cand in cands]
            filename = f'VisualDialog_test2018_{idx:012d}.jpg'

            cands = '<eos>'.join(cands)
            out = {'id': f"{idx}_{i}", 'image_id': f'{idx}', 'round_id': f'{i}',
                   'question': question, 'candidates': cands,
                   'filename': filename}
            res.append(out)

    print(f'size: ({len(res)})')

    image_dir = str(root / 'visdial/images/VisualDialog_test2018')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = out_dir / f'{clip_model_name}_test_vision.pkl'
    if not out_path.is_file():
        run_split(clip_model_type, 'test', image_dir, out_path, num_gpus, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
