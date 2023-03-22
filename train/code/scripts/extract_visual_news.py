import tarfile

from tqdm import tqdm

with tarfile.open("sample.tar.gz", "r:gz") as f
    for tarinfo in tqdm(f):
        if tarinfo.isreg():
