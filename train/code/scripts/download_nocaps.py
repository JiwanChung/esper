import re
import json
import shutil
import requests
from pathlib import Path

from tqdm import tqdm


ann_url = 'https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json'
out_path = '../../data/nocaps'
out_path = Path(out_path)
out_path.mkdir(exist_ok=True, parents=True)
img_path = out_path / 'images'
img_path.mkdir(exist_ok=True, parents=True)


def download(url, tgt):
    if not tgt.is_file():
        with requests.get(url, stream=True) as r:
            total_length = int(r.headers.get("Content-Length"))
            with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                with open(tgt, "wb") as f:
                    shutil.copyfileobj(raw, f)


ann_path = out_path / 'nocaps_val.json'
download(ann_url, ann_path)

with open(ann_path) as f:
    ann = json.load(f)
images = ann['images']

for image in tqdm(images, total=len(images)):
    url = image['coco_url']
    idx = image['id']
    suffix = Path(image['file_name']).suffix
    fname = f'{idx}{suffix}'
    download(url, img_path / fname)
