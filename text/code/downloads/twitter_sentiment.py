'''
http://help.sentiment140.com/for-students/
'''
import csv
import json
import shutil
import zipfile
from pathlib import Path

import gdown
from tqdm import tqdm

from utils import root


_DATA_URL = "https://drive.google.com/u/0/uc?id=0B04GJPshIjmPRnZManQwWEdTZjg&export=download"


zip_path = root / 'data/raw/twitter_sentiment.zip'
data_dir = root / 'data/raw/twitter_sentiment'
out_dir = root / 'data/texts'
zip_path.parent.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

data_path = data_dir / 'training.1600000.processed.noemoticon.csv'

if not data_path.is_file():
    if not zip_path.is_file():
        gdown.download(_DATA_URL, str(zip_path), quiet=False)
    with zipfile.ZipFile(str(zip_path), 'r') as f:
        f.extractall(str(data_dir))

data = []
try:
    with open(data_path) as f:
        reader = csv.reader(f)
        for row in tqdm(reader):
            txt = row[-1].strip()
            data.append(txt)
except:
    pass


res = data
print(f"{len(res)} lines in total")
with open(Path(out_dir) / 'twitter_sentiment.txt', 'w') as f:
    for line in res:
        f.write(f'{line}\n')
