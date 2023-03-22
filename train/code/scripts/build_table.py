import json
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

import numpy as np
from simple_parsing import ArgumentParser
from tqdm import tqdm


@dataclass
class Options:
    candidate_dir: str = '../../data/log/paper/human_eval'

parser = ArgumentParser()
parser.add_arguments(Options, dest="options")
args = parser.parse_args()
args = args.options

out_path = Path(args.candidate_dir).parent / 'table.json'

styles = ['sns', 'news', 'blog', 'instruction', 'story']
metrics = ['Bleu_4', 'METEOR', 'CIDEr']

candidates = list(Path(args.candidate_dir).glob('*.json'))
res = defaultdict(lambda: defaultdict(lambda: {}))
lengths = defaultdict(lambda: [])
totals = defaultdict(lambda: defaultdict(lambda: []))
for cand in tqdm(candidates):
    with open(cand) as f:
        data = json.load(f)
    stats = data['stats']
    name = cand.stem
    if name.startswith('clip_infer'):  # no full inference for clip infer
        continue
    if styles[0] not in stats:
        continue
    for style in styles:
        for metric in metrics:
            val = stats[style][metric]
            val = float(val)
            res[name][style][metric] = val
            totals[name][metric].append(val)
        lengths[style].append(stats[style]['length'])

for k, v in lengths.items():
    assert all(np.array(v) == v[0]), f'length diff in {k}'

for name, val in totals.items():
    for metric, val2 in val.items():
        res[name]['total'][metric] = np.array(val2).mean()

texts = {}
for name, v in res.items():
    text = ''
    for style in [*styles, 'total']:
        for metric in metrics:
            val = v[style][metric]
            val = val * 100  # max 100?
            text += f'& {val:.01f}'
    texts[name] = text

with open(out_path, 'w') as f:
    json.dump(texts, f, indent=4)
