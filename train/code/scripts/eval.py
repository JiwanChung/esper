import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from metric.lang_metrics import Eval


parser = argparse.ArgumentParser()
parser.add_argument(
    # '--src', type=str, default='../../data/style/style/generations/valid/size_3000/ours/model_inspects_prefix_dot_this_is_a_picture_of_a/essay', help='src file path')
    '--src', type=str, default='../../data/log/ppo/03-24-2022_15:24:45/generation.json', help='src file path')
parser.add_argument(
    '--tgt', type=str, default='../../data/coco/images/style_aws/100/result_batch1.json', help='tgt file path')
parser.add_argument(
    '--out-name', type=str, default='baseline', help='output filename')
args = parser.parse_args()

style_map = {
    'sns': 'twitter',
}

eval_f = Eval()

args.src = Path(args.src)
args.tgt = Path(args.tgt)
if args.src.is_dir():
    data = {}
    for p in args.src.glob('*.json'):
        with open(p) as f:
            x = json.load(f)
        data = {**data, **x}
else:
    with open(args.src) as f:
        data = json.load(f)

with open(args.tgt) as f:
    tgt = json.load(f)

res = defaultdict(lambda: [])
for k, row in tgt.items():
    filename = row['file_name']
    if filename in data:
        src_row = data[filename]
    else:
        fid = Path(filename).stem
        src_row = data[fid]
    for style, cap in row['stylecap'].items():
        cap = '.'.join(cap.split('.')[:2]).strip()  # get first 2 sentences
        if isinstance(src_row, str):
            out = [src_row, cap]
        else:
            map_style = style_map.get(style, style)
            if map_style in src_row:
                out = [src_row[map_style], cap]
            else:
                out = [src_row['corpus'], cap]
        res[style].append(out)


out = []
for style, li in res.items():
    res, gts = zip(*li)
    res = list(res)
    gts = list(gts)
    stats = eval_f(res, gts, cut=True)
    stats = {'style': style, **stats}
    out.append(stats)

df = pd.DataFrame(out)
df.to_csv(f'./{args.out_name}.csv')
