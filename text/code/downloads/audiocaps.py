'''
https://github.com/cdjkim/audiocaps/tree/master/dataset
'''
import json
import csv
import zipfile
from pathlib import Path

from tqdm import tqdm

from utils import root, download, download_base


_DATA_URL = "https://github.com/cdjkim/audiocaps/raw/master/dataset/train.csv"

data_dir = root / 'data/raw/audiocaps'
out_dir = root / 'data/texts'
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

data_path = data_dir / 'train.csv'


if not data_path.is_file():
    download_base(_DATA_URL, data_path)

captions = []
with open(data_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = 'caption'
        caption = row[name].strip()
        captions.append(caption)

print(f"{len(captions)} lines in total")
with open(Path(out_dir) / 'audio_caps.txt', 'w') as f:
    for line in captions:
        f.write(f'{line}\n')
