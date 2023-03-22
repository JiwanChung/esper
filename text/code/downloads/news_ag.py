'''
https://github.com/mhjabreel/CharCnn_Keras
'''
import re
import csv
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from utils import download_base, root


_DATA_URLS = [
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
]

data_dir = root / 'data/raw/news_ag'
out_dir = root / 'data/texts'
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)


regex = re.compile(r'\(.*\)$')
data = []
for url in _DATA_URLS:
    path = data_dir / Path(url).name
    if not path.is_file():
        download_base(url, path)

    with open(path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(
            csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
        )
        for id_, row in tqdm(enumerate(csv_reader), desc=path.stem):
            label, title, description = row
            label = int(label) - 1
            title = regex.sub('', title.strip()).strip()
            text = ": ".join((title, description.strip()))
            data.append(text)


print(f"{len(data)} lines in total")
with open(Path(out_dir) / f'news_ag.txt', 'w') as f:
    for line in data:
        f.write(f'{line}\n')
