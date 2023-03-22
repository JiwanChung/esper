'''
https://huggingface.co/datasets/blog_authorship_corpus/blob/main/blog_authorship_corpus.py
'''
import shutil
import zipfile
from pathlib import Path

import gdown
from tqdm import tqdm

from utils import root


_URL = "https://lingcog.blogspot.com/p/datasets.html"
_DATA_URL = "https://drive.google.com/u/0/uc?id=1cGy4RNDV87ZHEXbiozABr9gsSrZpPaPz&export=download"

zip_path = root / 'data/raw/blogs.zip'
data_dir = root / 'data/raw/blogs'
out_dir = root / 'data/texts'
zip_path.parent.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)


if not len(list(data_dir.glob('*.xml'))) > 0:
    if not zip_path.is_file():
        gdown.download(_DATA_URL, str(zip_path), quiet=False)
    with zipfile.ZipFile(str(zip_path), 'r') as f:
        f.extractall(str(data_dir.parent))
    shutil.rmtree(str(data_dir.parent / "__MACOSX"))
files = sorted(list(data_dir.glob('*.xml')))

res = []
for file_path in tqdm(files, total=len(files)):
    with open(file_path, encoding="latin_1") as f:
        date = ""
        for line in f:
            line = line.strip()
            if line and not line.startswith('<'):
                res.append(line.strip().replace('\n', '<LB>'))

print(f"{len(res)} lines in total")
with open(Path(out_dir) / 'blog.txt', 'w') as f:
    for line in res:
        f.write(f'{line}\n')
