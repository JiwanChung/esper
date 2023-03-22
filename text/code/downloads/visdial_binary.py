'''
https://visualdialog.org/
'''
import json
import random
import csv
import zipfile
from itertools import chain
from pathlib import Path

from tqdm import tqdm

from utils import root, download


_DATA_URL = "https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.9_train.zip?dl=1"

seed = 0
zip_path = root / 'data/raw/visdial_1.9_train.zip'
data_dir = root / 'data/raw/visdial_binary'
out_dir = root / 'data/texts'
zip_path.parent.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

data_path = data_dir / 'visdial_1.0_train.json'


if not data_path.is_file():
    if not zip_path.is_file():
        download(_DATA_URL, zip_path)
    with zipfile.ZipFile(str(zip_path), 'r') as f:
        f.extractall(str(data_dir))

with open(data_path) as f:
    data = json.load(f)

questions = data['data']['questions']
answers = data['data']['answers']

def get_row(turn):
    que = questions[turn['question']].strip()
    gt_idx = turn['gt_index']
    options = turn['answer_options']
    rands = list(range(len(options)))
    rands.pop(gt_idx)
    fake_idx = random.choice(rands)
    true_ans = answers[options[gt_idx]].strip()
    false_ans = answers[options[fake_idx]].strip()
    true_text = f"A: {que}\nB: {true_ans}\nA: True"
    false_text = f"A: {que}\nB: {false_ans}\nA: False"
    return true_text, false_text


random.seed(seed)
dialogues = []
data = data['data']['dialogs']
for dial in tqdm(data, total=len(data)):
    dial = list(chain(*[get_row(turn) for turn in dial['dialog']]))
    dial = [row.replace('\n', '<|LB|>') for row in dial]
    dialogues.extend(dial)
data = dialogues

print(f"{len(data)} lines in total")
with open(Path(out_dir) / 'visdial_binary.txt', 'w') as f:
    for line in data:
        f.write(f'{line}\n')
