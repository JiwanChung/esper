import csv
import json
from pathlib import Path

from tqdm import tqdm

from utils import download_base, root

_DATA_URL = 'https://github.com/snigdhac/StoryComprehension_EMNLP/blob/master/Dataset/RoCStories/100KStories.csv'

data_dir = root / 'data/raw/story'
out_dir = root / 'data/texts'
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

path = data_dir / 'story.csv'

with open('env.json') as f:
    config = json.load(f)

_DATA_URL = config['story_path']

download_base(_DATA_URL, path)


def remove_quotation(txt):
    txt = txt.strip()
    if txt.startswith('"') and txt.endswith('"') and len(txt) >= 2:
        txt = txt[1:-1]
    if txt.startswith("'") and txt.endswith("'") and len(txt) >= 2:
        txt = txt[1:-1]
    return txt


res = []
with open(path) as f:
    csvreader = csv.reader(f)
    header = next(csvreader)
    for row in csvreader:
        res.append(row)


print(f"{len(res)} lines in total")
with open(Path(out_dir) / f'story.txt', 'w') as f:
    for line in res:
        txt = ' '.join(line[2:])
        f.write(f'{txt}\n')
