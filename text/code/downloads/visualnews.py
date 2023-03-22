'''
https://github.com/FuxiaoLiu/VisualNews-Repository

given access to the dataset, download and extract to get the 'data.json' file.
'''
import json
import random
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from utils import root


seed = 1

random.seed(seed)
data_path = root / 'data/raw/visualnews/data.json'
out_dir = root / 'data/texts'
out_dir.mkdir(exist_ok=True)

with open(data_path) as f:
    x = json.load(f)

tqdm.write('formatting data')
sources = defaultdict(lambda: [])
for row in tqdm(x, total=len(x), desc='splitting sources'):
    source = row['source']
    sources[source].append(row)

split_sizes = {
    'train': 100000,
    # 'val': 10000,
    # 'test': 10000
}
all_data = defaultdict(lambda: [])
num_sources = len(sources)
all_sizes = sum(split_sizes.values())
for source, data in tqdm(sources.items(), total=len(sources), desc='getting splits'):
    samples = random.sample(data, all_sizes)
    ptr = 0
    for name, size in split_sizes.items():
        split = samples[ptr: ptr+size]
        all_data[source].extend(split)
        all_data['news'].extend(split)
        ptr += size

for source, v in all_data.items():
    print(f'size: {source} ({len(v)})')

for source, data in all_data.items():
    data = [v['caption'].strip().replace('\n', '<|LB|>') for v in data]
    with open(Path(out_dir) / f'{source.lower()}_visualnews.txt', 'w') as f:
        for line in data:
            f.write(f'{line}\n')
