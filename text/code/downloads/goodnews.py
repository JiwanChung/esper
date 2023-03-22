'''
https://github.com/furkanbiten/GoodNews
'''
import json
from pathlib import Path
import shutil
import subprocess

from tqdm import tqdm

from utils import download_base, root


data_dir = root / 'data/raw/goodnews'
out_dir = root / 'data/texts'
file_path = data_dir / 'news_dataset.json'
#split_path = data_dir / 'splits.json'

# shutil.rmtree(str(data_dir))
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

# if not (file_path.is_file() and split_path.is_file()):
if not (file_path.is_file()):
    shutil.copyfile('./download_goodnews.sh', str(data_dir / 'download.sh'))
    subprocess.run(['bash', str(data_dir / 'download.sh')])

with open(file_path) as f:
    x = json.load(f)

'''
with open(split_path) as f:
    splits = json.load(f)
'''

data = []
for row in tqdm(x):
    if row['split'] == 'train':
        # caption = row['sentences'][0]['raw']
        caption = row['sentences_full'][0]['raw']
        data.append(caption.strip())

print(f"{len(data)} lines in total")
with open(Path(out_dir) / f'goodnews.txt', 'w') as f:
    for line in data:
        f.write(f'{line}\n')
