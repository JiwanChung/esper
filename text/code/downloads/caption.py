'''
'''
import json
import csv
import zipfile
from pathlib import Path

from tqdm import tqdm

from utils import root, download


_DATA_URL = "http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"

zip_path = root / 'data/raw/caption.zip'
data_dir = root / 'data/raw/caption'
out_dir = root / 'data/texts'
zip_path.parent.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

data_path = data_dir / 'dataset_coco.json'


if not data_path.is_file():
    if not zip_path.is_file():
        download(_DATA_URL, zip_path)
    with zipfile.ZipFile(str(zip_path), 'r') as f:
        f.extractall(str(data_dir))

with open(data_path) as f:
    data = json.load(f)

captions = []
for img in data['images']:
    for sent in img['sentences']:
        captions.append(sent['raw'].strip())
data = captions

print(f"{len(data)} lines in total")
with open(Path(out_dir) / 'caption.txt', 'w') as f:
    for line in data:
        f.write(f'{line}\n')
