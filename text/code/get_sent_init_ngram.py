import json
import requests
import math
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
from simple_parsing import ArgumentParser


@dataclass
class Config:
    ngram: int = 2
    data_path: str = '../data/texts'


def download_base(url, tgt):
    if not tgt.is_file():
        x = requests.get(url)
        with open(tgt, 'wb') as f:
            f.write(x.content)


def parse_args():
    parser = ArgumentParser()
    parser.add_arguments(Config, dest='config')
    args = parser.parse_args()
    args = args.config
    return args


def main():
    args = parse_args()
    args.name = f'sent_init_{args.ngram}gram'
    url = f'https://www.ngrams.info/coca/samples/coca_ngrams_x{args.ngram}w.txt'
    root = Path(args.data_path)
    raw_path = root / f'{args.name}.txt'
    if not raw_path.is_file():
        download_base(url, raw_path)
    txts = []
    nums = []
    with open(raw_path) as f:
        for line in f:
            line = line.split('\t')
            try:
                num = int(line[0])
            except:
                continue
            txt = ' '.join(line[1:]).strip()
            nums.append(float(num))
            txts.append(txt)

    nums = np.array(nums)
    nums = nums / nums.sum()
    res = dict(zip(txts, nums.tolist()))
    out_path = root / f'{args.name}.json'
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=4)
    print('done')


if __name__ == '__main__':
    main()
