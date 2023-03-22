import os
import random
import json
import math
import logging
from pathlib import Path
from itertools import chain
from dataclasses import dataclass

from simple_parsing import ArgumentParser
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import clip

from metric.lang_metrics import Eval
from data import (
    ClipCocoCollator, ClipCocoDataset
)


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


@dataclass
class Config:
    clip_model_type: str = 'ViT-B/32'
    seed: int = 1
    batch_size: int = 1024
    num_workers: int = 8
    num_chunks: int = 4
    topk: int = 5
    out_path: str = '../data/coco/cache/coco_clip_retrieval_train.json'
    debug: bool = False


parser = ArgumentParser()
parser.add_arguments(Config, dest='config')
args = parser.parse_args().config

assert Path(args.out_path).parent.is_dir()

random.seed(args.seed)

device = 'cuda'

log.info('loading data')
dataset = ClipCocoDataset(model_name='gpt2',
                          split='val' if args.debug else 'train',
                          clip_model_type=args.clip_model_type,
                          use_caption=True,
                          use_coco_eval=True,
                          use_label_prefix=False)
tokenizer = dataset.tokenizer
collator = ClipCocoCollator(tokenizer=tokenizer)

dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False,
                        drop_last=False, collate_fn=collator)

log.info('loading model')
clip_model, _ = clip.load(args.clip_model_type, device=device, jit=False)

log.info('extracting feature')
image_keys = []
gt_texts = []
image_feats = []
text_keys = []  # for accuracy
text_feats = []
text_raws = []
with torch.no_grad():
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        text = list(chain(*batch['coco_captions']))
        gt_texts.extend(batch['coco_captions'])
        image = batch['features']
        text_ = clip.tokenize(text, truncate=True).to(device)
        feat = clip_model.encode_text(text_)
        feat /= feat.norm(dim=-1, keepdim=True)
        image /= image.norm(dim=-1, keepdim=True)
        feat = feat.cpu().numpy()
        image = image.cpu().numpy()
        text_raws.extend(text)
        text_feats.append(feat)
        image_keys.extend(batch['image_ids'])
        # tk = list(chain(*[[i for _ in range(5)] for i in batch['image_ids']]))
        tk = [[i for _ in c] for c, i in zip(batch['coco_captions'], batch['image_ids'])]
        tk = list(chain(*tk))
        text_keys.extend(tk)
        image_feats.append(image)

log.info('concat')

text_feats = list(chain(*[list(v) for v in text_feats]))

# roll
shuffle_ids = list(range(len(text_keys)))
random.shuffle(shuffle_ids)
text_keys = [text_keys[i] for i in shuffle_ids]
text_raws = [text_raws[i] for i in shuffle_ids]
text_feats = [text_feats[i] for i in shuffle_ids]

text_feats = np.stack(text_feats, axis=0)
image_feats = np.concatenate(image_feats, axis=0)

log.info(f'shape: (text/{len(text_feats)}) (image/{len(image_feats)})')

with torch.no_grad():
    log.info('calc sim')
    sims = []
    for text_i in tqdm(range(args.num_chunks), total=args.num_chunks):
        sub_sims = []
        chunk_size = math.ceil(len(text_feats) / args.num_chunks)
        sub_text = text_feats[chunk_size * text_i: chunk_size * (text_i + 1)]
        sub_text = torch.from_numpy(sub_text).to(device)
        for image_i in range(args.num_chunks):
            ichunk_size = math.ceil(len(image_feats) / args.num_chunks)
            sub_image = image_feats[ichunk_size * image_i: ichunk_size * (image_i + 1)]
            sub_image = torch.from_numpy(sub_image).to(device)
            subsub_sim = torch.einsum('ic,tc->it', sub_image, sub_text)
            sub_sims.append(subsub_sim.cpu())
        sub_sims = torch.cat(sub_sims, dim=0)
        sims.append(sub_sims)
    sims = torch.cat(sims, dim=1)

    torch.cuda.empty_cache()

    log.info('get topk')
    text_raws = np.array(text_raws)
    topk = []
    for i in tqdm(range(args.num_chunks), total=args.num_chunks):
        chunk_size = math.ceil(len(sims) / args.num_chunks)
        sub_sim = sims[chunk_size * i: chunk_size * (i + 1)]
        sub_topk = sub_sim.to(device).topk(k=args.topk, dim=1, largest=True, sorted=True)
        sub_topk = sub_topk.indices.cpu().numpy()
        topk.append(sub_topk)
    topk = np.concatenate(topk, axis=0)
    text_topk = {k: list(text_raws[v]) for k, v in zip(image_keys, topk)}

# acc
text_keys = np.array(text_keys)
image_check = [float(k == text_keys[v][0]) for k, v in zip(image_keys, topk)]
image_acc = np.array(image_check).mean()
log.info(f'r@1: {image_acc}')
image_check = [float(any([k == k2 for k2 in text_keys[v]])) for k, v in zip(image_keys, topk)]
image_acc = np.array(image_check).mean()
log.info(f'r@{args.topk}: {image_acc}')

assert len(text_topk) == len(image_feats)
sample = text_topk[list(text_topk.keys())[0]]
assert len(sample) == args.topk
assert isinstance(sample[0], str)

hypo = {k: v[0] for k, v in text_topk.items()}
tgt = dict(zip(image_keys, gt_texts))
keys = list(tgt.keys())
hypo = [hypo[k] for k in keys]
tgt = [tgt[k] for k in keys]
metrics = Eval()
stats = metrics(hypo, tgt)
print(stats)

log.info('saving')
with open(args.out_path, 'w') as f:
    json.dump(text_topk, f, indent=4)

log.info('done')
