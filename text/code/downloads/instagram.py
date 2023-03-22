'''
https://github.com/karansikka1/documentIntent_emnlp19
'''
import json
from itertools import chain
from pathlib import Path
from collections import defaultdict

from utils import download_base, root


_DATA_URL = "https://raw.githubusercontent.com/karansikka1/documentIntent_emnlp19/master/splits"

data_dir = root / 'data/raw/instagram'
out_dir = root / 'data/texts'
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

files = [
    *[f'train_split_{i}.json' for i in range(5)],
    *[f'val_split_{i}.json' for i in range(5)]
]

for f in files:
    path = data_dir / f
    url = f'{_DATA_URL}/{f}'
    download_base(url, path)


def load_file(p):
    with open(p, 'r') as f:
        res = json.load(f)
    return res


data = [load_file(data_dir / p) for p in files]
data = list(chain(*data))

print(f"{len(data)} lines in total")
with open(Path(out_dir) / f'instagram.txt', 'w') as f:
    for line in data:
        text = line['orig_caption'].strip()
        f.write(f'{text}\n')
