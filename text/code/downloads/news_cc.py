'''
https://commoncrawl.org/2016/10/news-dataset-available/
'''
import json
import re
import shutil
import csv
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import gdown

from utils import download_base, root


_DATA_URL = "https://storage.googleapis.com/huggingface-nlp/datasets/cc_news/cc_news.tar.gz"


data_dir = root / 'data/raw/news'
out_dir = root / 'data/texts'
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)
zip_path = data_dir / Path(_DATA_URL).name
data_path = data_dir / 'cc_download_articles'


regex = re.compile(r'\(.*\)$')
data = []
if not data_path.is_dir():
    if not zip_path.is_file():
        # download_base(_DATA_URL, zip_path)
        gdown.download(_DATA_URL, str(zip_path), quiet=False)

    shutil.unpack_archive(str(zip_path), str(data_dir))

files = list(data_path.rglob('*.json'))
data = []
skipped = 0
for p in tqdm(files, total=len(files)):
    with open(p) as f:
        x = json.load(f)
        if 'title' in x and x['title'] and 'description' in x and x['description']:
            title = x['title'].strip()
            desc = x['description'].strip()
            text = f'{title}: {desc}'
            data.append(text)


print(f"{len(data)} lines in total")
print(f"{skipped} lines skipped")
with open(Path(out_dir) / f'news.txt', 'w') as f:
    for line in data:
        f.write(f'{line}\n')
