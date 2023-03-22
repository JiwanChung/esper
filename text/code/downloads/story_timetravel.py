'''
https://github.com/qkaren/Counterfactual-StoryRW
'''
import json
import shutil
import zipfile
from pathlib import Path

import gdown
from tqdm import tqdm

from utils import root


_DATA_URL = "https://drive.google.com/u/0/uc?id=150jP5FEHqJD3TmTO_8VGdgqBftTDKn4w&export=download"


zip_path = root / 'data/raw/story_timetravel.zip'
data_dir = root / 'data/raw/story_timetravel'
out_dir = root / 'data/texts'
zip_path.parent.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

data_path = data_dir / 'TimeTravel' / 'train_supervised_large.json'

if not data_path.is_file():
    if not zip_path.is_file():
        gdown.download(_DATA_URL, str(zip_path), quiet=False)
    with zipfile.ZipFile(str(zip_path), 'r') as f:
        f.extractall(str(data_dir))
    shutil.rmtree(str(data_dir / "__MACOSX"))

data = []
with open(data_path) as f:
    for line in f:
        data.append(json.loads(line.strip()))

res = []
for x in tqdm(data, total=len(data)):
    row = ' '.join([x['premise'], x['initial'], x['original_ending']])
    res.append(row)
    for end in x['edited_ending']:
        row = ' '.join([x['premise'], x['counterfactual'], end])
        res.append(row)

print(f"{len(res)} lines in total")
with open(Path(out_dir) / 'story_timetravel.txt', 'w') as f:
    for line in res:
        f.write(f'{line}\n')
