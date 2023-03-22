import shutil
import requests
from pathlib import Path

from tqdm import tqdm


_DATA_URL = "http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"


def download(url, tgt):
    if not tgt.is_file():
        with requests.get(url, stream=True) as r:
            total_length = int(r.headers.get("Content-Length"))
            with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                with open(tgt, "wb") as f:
                    shutil.copyfileobj(raw, f)


def download_annotations():
    zip_path = Path('../data/coco/captions/captions.zip')
    data_dir = zip_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / 'dataset_coco.json'

    if not data_path.is_file():
        if not zip_path.is_file():
            download(_DATA_URL, zip_path)
        with zipfile.ZipFile(str(zip_path), 'r') as f:
            f.extractall(str(data_dir))

    return data_path
