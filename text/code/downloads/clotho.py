'''
https://zenodo.org/record/3490684#.YlVAndNBxhF
'''
import json
import csv
import zipfile
from pathlib import Path

from tqdm import tqdm

from utils import root, download


_DATA_URL = "https://zenodo.org/record/3490684/files/clotho_captions_development.csv"

data_dir = root / 'data/raw/clotho'
out_dir = root / 'data/texts'
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

data_path = data_dir / 'clotho_captions_development.csv'


if not data_path.is_file():
    download(_DATA_URL, data_path)

captions = []
with open(data_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        for i in range(1, 6):
            name = f'caption_{i}'
            caption = row[name].strip()
            captions.append(caption)

print(f"{len(captions)} lines in total")
with open(Path(out_dir) / 'audio_clotho.txt', 'w') as f:
    for line in captions:
        f.write(f'{line}\n')
