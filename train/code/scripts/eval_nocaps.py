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
    ann_path: str = '../../data/nocaps/nocaps_val.json'
    prompt: str = ''

parser = ArgumentParser()
parser.add_arguments(Options, dest="options")
args = parser.parse_args()
args = args.options


with open(args.ann_path) as f:
    ann = json.load(f)
domains = {v['id']: v['domain'] for v in ann['images']}
ann = ann['annotations']

tgt = {}
res = defaultdict(lambda: defaultdict(lambda: []))
for row in ann:
    domain = domains[row['image_id']]
    idx = row['image_id']
    idx = f'{idx}'
    res[domain][idx].append(row['caption'])
    res['overall'][idx].append(row['caption'])

eval_ = Eval(use_spice=True)

name = Path(args.file_path).name
with open(args.file_path) as f:
    hypo = json.load(f)

stats = {}
for domain, tgt in res.items():
    keys = list(tgt.keys())
    hypo_sub = [hypo[key][len(args.prompt):].strip() for key in keys]
    tgt_sub = [tgt[key] for key in keys]
    metrics = eval_(hypo_sub, tgt_sub)
    metrics['num'] = len(hypo_sub)
    stats[domain] = metrics

print(f'---{name}---')
print(json.dumps(stats, indent=4))
