import json
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

import numpy as np
from tqdm import tqdm
from simple_parsing import ArgumentParser

from metric.lang_metrics import Eval


@dataclass
class Options:
    file_path: str = '../../data/temp/gen/visualnews_zs.json'
    file_dir: Optional[str] = None
    ann_path: str = '../../data/visualnews/origin/data.json'
    prompt: str = ''

parser = ArgumentParser()
parser.add_arguments(Options, dest="options")
args = parser.parse_args()
args = args.options


with open(args.ann_path) as f:
    ann = json.load(f)

res = defaultdict(lambda: defaultdict(lambda: []))
for row in tqdm(ann, total=len(ann), desc='formatting'):
    domain = row['source']
    idx = row['image_path']
    res[domain][idx].append(row['caption'])
    res['overall'][idx].append(row['caption'])

eval_ = Eval(use_spice=False)

name = Path(args.file_path).name
with open(args.file_path) as f:
    hypo = json.load(f)['generations']

stats = {}
for domain, tgt in res.items():
    domain = domain.split('_')[0]
    if domain == 'overall':
        hypo_sub = hypo
    else:
        hypo_sub = {k: v for k, v in hypo.items() if domain in v}
    keys = list(set(hypo_sub.keys()) & set(tgt.keys()))
    name = domain
    if domain == 'overall':
        name = 'news'
    hypo_sub = [hypo_sub[key][name][len(args.prompt):].strip() for key in keys]
    tgt_sub = [tgt[key] for key in keys]
    metrics = eval_(hypo_sub, tgt_sub)
    metrics['num'] = len(hypo_sub)
    x = list(zip(hypo_sub, tgt_sub))
    stats[domain] = metrics


means = defaultdict(lambda: [])
for domain, stat in stats.items():
    if domain == 'overall':
        continue
    for k, v in stat.items():
        means[k].append(v)

means = {k: sum(v) if k == 'num' else np.array(v).mean() for k, v in means.items()}
stats['style'] = means


print(f'---{name}---')
print(json.dumps(stats, indent=4))
