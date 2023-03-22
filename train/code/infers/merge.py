import json
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np


def run_merge(args, run):
    all_stats = defaultdict(lambda: [])
    for i, ckpt in enumerate(args.infer_multi_checkpoint):
        print(f'ckpt {i}: {ckpt}')
        subargs = copy.deepcopy(args)
        subargs.checkpoint = str(Path(ckpt).resolve())
        stats = run(subargs)
        for k, v in stats.items():
            all_stats[k].append(float(v))

    stats = {}
    for k, v in all_stats.items():
        v = np.array(v)
        row = {
            'mean': v.mean(),
            'std': v.std()
        }
        stats[k] = row

    print(json.dumps(stats, indent=4))

    with open(args.infer_out_path, 'w') as f:
        json.dump(stats, f, indent=4)
