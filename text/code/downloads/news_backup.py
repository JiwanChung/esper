'''
http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
'''
import bz2
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from utils import download, root


_DATA_URL = "http://groups.di.unipi.it/~gulli/newsSpace.bz2"

data_dir = root / 'data/raw/news'
out_dir = root / 'data/texts'
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

zip_path = data_dir / Path(_DATA_URL).name
db_path = data_dir / Path(_DATA_URL).name[:-4]

download(_DATA_URL, zip_path)

if not db_path.is_file():
    with open(db_path, 'wb') as new_file, bz2.BZ2File(str(zip_path), 'rb') as file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            new_file.write(data)

head = True
end = False
res = []
sents = []
with open(db_path, 'rb') as f:
    for line in tqdm(f):
        line = str(line)
        line = line.strip()
        if head:
            line = line.split('\t')[-1].strip()
            line = '-'.join(line.split('-')[1:])
            head = False
        else:
            line = line[2::]
        line = line[:-3]
        if line.endswith(r'\N'):
            head = True
            end = True
            line = line.split(r'\t')[0]
        else:
            line = line[:-2]
        line = line.strip()
        sents.append(line)
        if end:
            res.append(' '.join(sents))
            sents = []
            end = False


print(f"{len(res)} lines in total")
with open(Path(out_dir) / f'news.txt', 'w') as f:
    for line in res:
        f.write(f'{line}\n')
