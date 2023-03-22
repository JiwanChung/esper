'''
https://github.com/cardiffnlp/tweeteval
'''
from pathlib import Path
from collections import defaultdict

from utils import download, root


_DATA_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion"

data_dir = root / 'data/raw/twitter'
out_dir = root / 'data/texts'
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

files = [
    "mapping.txt",
    "train_labels.txt",
    "train_text.txt",
    "test_labels.txt",
    "test_text.txt",
    "val_labels.txt",
    "val_text.txt",
]

for f in files:
    url = f'{_DATA_URL}/{f}'
    download(url, data_dir / f)


def load_file(p):
    res = []
    with open(p, 'r') as f:
        for line in f:
            res.append(line.strip())
    return res

mapping = load_file(data_dir / 'mapping.txt')
mapping = dict([v.split('\t') for v in mapping])

res = defaultdict(lambda: [])
for split in ['train', 'val', 'test']:
    text = load_file(data_dir / f'{split}_text.txt')
    labels = load_file(data_dir / f'{split}_labels.txt')
    for x, y in zip(text, labels):
        res[mapping[y]].append(x)

for k, v in res.items():
    print(f"{k.lower()}: {len(v)} lines in total")
    with open(Path(out_dir) / f'twitter_{k.lower()}.txt', 'w') as f:
        for line in v:
            f.write(f'{line}\n')
