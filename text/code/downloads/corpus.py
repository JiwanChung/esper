'''
https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
'''
import zipfile
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from utils import download, root


_DATA_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"

zip_path = root / 'data/raw/corpus.zip'
data_dir = root / 'data/raw/corpus'
out_dir = root / 'data/texts'
zip_path.parent.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

if not zip_path.is_file():
    download(_DATA_URL, zip_path)
with zipfile.ZipFile(str(zip_path), 'r') as f:
    f.extractall(str(data_dir))

data_path = data_dir / 'wikitext-103-raw'
paths = sorted(list(data_path.glob('wiki.*.raw')))
res = []
for p in paths:
    with open(p) as f:
        for line in tqdm(f):
            line = line.strip()
            if line and not line.startswith('='):
                line = line.replace(' .', '.')
                line = line.replace(' ,', ',')
                line = line.replace(' (', '(')
                line = line.replace(' )', ')')
                line = line.replace(' [', '[')
                line = line.replace(' ]', ']')
                line = line.replace(' :', ':')
                line = line.replace(' ;', ';')
                line = line.replace(' "', '"')
                line = line.replace(" '", "'")
                res.append(line)


print(f"{len(res)} lines in total")
with open(Path(out_dir) / 'corpus.txt', 'w') as f:
    for line in res:
        f.write(f'{line}\n')
