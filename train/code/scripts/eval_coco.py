import json
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

from simple_parsing import ArgumentParser

from metric.lang_metrics import Eval


@dataclass
class Options:
    file_path: str = '../../data/log/baselines/generations/sample.json'
    file_dir: Optional[str] = None
    ann_path: str = '../../data/coco/captions/dataset_coco.json'
    prompt: str = ''

parser = ArgumentParser()
parser.add_arguments(Options, dest="options")
args = parser.parse_args()
args = args.options


with open(args.ann_path) as f:
    ann = json.load(f)
ann = ann['images']

tgt = {}
for row in ann:
    split = row['split']
    if split == 'test':
        sents = [sent['raw'].strip() for sent in row['sentences']]
        idx = f"{row['imgid']}"
        tgt[idx] = sents

eval_ = Eval()

if args.file_dir is None:
    file_paths = [args.file_path]
else:
    file_paths = list(Path(args.file_dir).glob('*.json'))

cider_map = {}
for path in file_paths:
    name = Path(path).name
    with open(path) as f:
        hypo = json.load(f)

    keys = list(hypo.keys())
    hypo = [hypo[key][len(args.prompt):].strip() for key in keys]
    tgt_sub = [tgt[key] for key in keys]
    metrics = eval_(hypo, tgt_sub)

    metrics['num'] = len(hypo)

    print(f'---{name}---')
    print(json.dumps(metrics, indent=4))
    cider_map[name] = metrics['CIDEr']

cider_li = reversed(sorted(list(cider_map.items()), key=lambda x: x[1]))
for line in cider_li:
    print(line)
