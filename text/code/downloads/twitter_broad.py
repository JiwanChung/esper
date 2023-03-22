'''
https://github.com/GateNLP/broad_twitter_corpus
'''
import json
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from utils import download_base, wikitext_detokenizer, root


_DATA_URL = "https://raw.githubusercontent.com/GateNLP/broad_twitter_corpus/master"

data_dir = root / 'data/raw/twitter_broad'
out_dir = root / 'data/texts'
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)


files = [
    "a.json",
    "b.json",
    "e.json",
    "f.json",
    "g.json"
]

data = []
for f in tqdm(files, total=len(files), desc='files'):
    url = f'{_DATA_URL}/{f}'
    path = data_dir / f
    download_base(url, path)
    with open(path) as f:
        for line in tqdm(f, desc='line'):
            x = json.loads(line.strip())
            x = x['text'].strip()
            x = wikitext_detokenizer(x)
            data.append(x)


print(f"{len(data)} lines in total")
with open(Path(out_dir) / f'twitter_broad.txt', 'w') as f:
    for line in data:
        f.write(f'{line}\n')
