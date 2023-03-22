import os
import json
import pickle
import random
import logging
import itertools
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import clip
import yaml
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from arguments import get_args
from policy import Policy
from ref_policy import RefPolicy
from utils.utils import get_first_sentence
from .common import load_model_with_args
from .nocaps import NocapsInferDataset


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

split = 'val'
use_ref = False

def reduce_sum(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask)
    return torch.sum(value * mask, axis)


def reduce_mean(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask) / torch.sum(mask)
    return reduce_sum(value, mask, axis) / torch.sum(mask, axis)


class VisDialInferDataset(NocapsInferDataset):
    def load_data(self):
        data_dir = Path('../data/visdial/cache/clipcap')
        with open(data_dir / f'{self.clip_model_name}_{split}_vision.pkl', 'rb') as f:
            all_data = pickle.load(f)
        prefixes = all_data["clip_embedding"].reshape(-1, 512)
        captions_raw = all_data["captions"]

        captions_raw = {v['id']: v for v in captions_raw}
        self.prefixes = prefixes
        self.captions_raw = captions_raw

    def get_feature(self, image_id: str, round_id):
        idx = f'{image_id}_{round_id}'
        if idx not in self.captions_raw:
            return None, None
        dt = self.captions_raw[idx]
        return dt, self.prefixes[dt['clip_embedding']]

    def getitem(self, image_id: str, round_id):
        dt, feature = self.get_feature(image_id, round_id)
        if dt is None:
            return None
        candidates = dt['candidates'].split('<eos>')
        question = dt['question']
        # question = f'dialogue: A: {question}\n'
        # candidates = [f'B: {v.strip()}' for v in candidates]
        res = {
            'image_id': int(dt['image_id']),
            'round_id': int(dt['round_id']),
            'question': question,
            'candidates': candidates,
            'feature': feature
        }
        return res


class VisDialCollator:
    def __init__(self, tokenizer, bl):
        self.tokenizer = tokenizer
        self.ans_idx_true = self.tokenizer.encode(' True')[0]
        self.bl = bl

    def __call__(self, sequences):
        sequences = [seq for seq in sequences if seq is not None]
        image_ids = [sequence['image_id'] for sequence in sequences]
        round_ids = [sequence['round_id'] for sequence in sequences]
        features = [sequence['feature'] for sequence in sequences]
        features = torch.stack(features, dim=0)
        questions = [sequence['question'] for sequence in sequences]

        candidates = [sequence['candidates'] for sequence in sequences]

        queries = ['dialogue: A: {q}\n' for q in questions]
        encodings_dict = self.tokenizer(queries, return_tensors="pt", padding=True)
        query_input_ids = encodings_dict['input_ids']
        query_mask = encodings_dict['attention_mask']

        candidates = list(zip(*candidates))  # 100 B
        response_input_ids = []
        response_mask = []
        answer_poses = []
        for cands in candidates:
            if self.bl:
                cands = [f'B:{c.strip()}\nA: True' for q,c in zip(questions, cands)]
            else:
                cands = [f'B:{c.strip()}' for q,c in zip(questions, cands)]
            encodings_dict = self.tokenizer(cands, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids']
            mask = encodings_dict['attention_mask']
            answer_mask = input_ids == self.ans_idx_true
            answer_pos = answer_mask.nonzero()[:, 1]
            response_input_ids.append(input_ids)
            response_mask.append(mask)
            answer_poses.append(answer_pos)

        batch = {
            'image_ids': image_ids,
            'round_ids': round_ids,
            'query_input_ids': query_input_ids,
            'query_mask': query_mask,
            'answer_pos': answer_poses,
            'response_input_ids': response_input_ids,
            'response_mask': response_mask,
            'features': features,
        }
        return batch


def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def prepare(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args.label_path = Path(args.label_path)
    args.infer_dir = Path(args.infer_dir)
    # assert args.label_path.is_file(), f"no label file: {args.label_path}"
    # assert args.infer_dir.is_dir(), f"no inference directory: {args.infer_dir}"

    args.response_length = 200
    args, model = load_model_with_args(args, device)

    labels = None
    args.use_labels = args.label_path != Path('None')
    if args.use_labels:
        with open(args.label_path) as f:
            labels = json.load(f)
        labels = list(labels.keys())
    data = VisDialInferDataset(args.ref_model, args.clip_model_type, args.use_caption,
                        fixed_prompt=args.fixed_prompt,
                        label_path=args.label_path,
                        use_label_prefix=args.use_label_prefix)
    collator = VisDialCollator(data.tokenizer, bl=args.visdial_binary_loss)

    def get_batch(ids):
        rounds = list(range(10))  # DEBUG
        batch = list(itertools.product(ids, rounds))
        meta = batch
        batch = [data.getitem(*v) for v in batch]
        batch = collator(batch)
        return meta, batch

    return args, model, data.tokenizer, data, get_batch, device


def main(args, sanity_check=False, infer_f=None):
    print("inference")
    # assert infer_f is not None, "please provide an inference function"
    args, model, tokenizer, data, get_batch, device = prepare(args)
    ref_policy = RefPolicy(model_name=args.ref_model, temperature=args.temperature,
                           device=device, model_weight=None)
    args.batch_size = min(1, args.batch_size // 1)  # 10 rounds in val, 100 answer candidates

    ann_path = f'../data/visdial/visdial_1.0_{split}.json'
    with open(ann_path) as f:
        ann = json.load(f)
    temp = {}
    image_ids = []
    for row in ann['data']['dialogs']:
        iid = row['image_id']
        image_ids.append(iid)
        for i, v in enumerate(row['dialog']):
            if 'answer_options' in v:
                idx = f'{iid}_{i}'
                temp[idx] = v
    ann = temp
    image_ids = list(set(image_ids))

    chunks = list(get_chunks(image_ids, args.batch_size))

    out = defaultdict(lambda: {})
    add_prompt = not args.use_labels

    results = []
    mean_ranks = []
    rat1s = []
    rat5s = []
    rat10s = []
    with torch.no_grad():
        for j, chunk in tqdm(enumerate(chunks), total=len(chunks), desc='iters'):
            meta, batch = get_batch(chunk)
            qids = batch['query_input_ids'].to(device)
            qmask = batch['query_mask'].to(device)
            features = batch['features'].to(device)
            log_probs = []
            log_ratios = []
            for i, (cand_ids, cand_mask, cand_pos) in tqdm(enumerate(zip(batch['response_input_ids'],
                                                               batch['response_mask'],
                                                               batch['answer_pos'])),
                                                 desc='candidates'):
                cand_ids = cand_ids.to(device)
                cand_mask = cand_mask.to(device)
                out = model.forward_pass(qids, qmask, cand_ids, cand_mask, features=features)
                if args.visdial_normalized_loss:
                    log_prob = out['response/pos_logit']
                else:
                    log_prob = out['response/log_prob']
                if args.visdial_binary_loss:
                    ids = cand_pos.to(log_prob.device)  # answer T/F position indices
                    log_prob = log_prob.gather(1, ids[:, None]).squeeze(1)
                else:
                    log_prob = reduce_sum(log_prob, cand_mask, -1).cpu()
                log_probs.append(log_prob)
            log_probs = torch.stack(log_probs, dim=1)  # B 100
            ordered = torch.argsort(log_probs, dim=1, descending=True)
            ranks = ordered.argsort(dim=1).detach().cpu().numpy().tolist()
            for iid, rid, rank, log_prob in zip(batch['image_ids'],
                                            batch['round_ids'],
                                            ranks,
                                            log_probs):
                idx = f'{iid}_{rid}'
                row = {
                    'image_id': iid,
                    'round_id': rid + 1,  # rounds start from 1
                    'ranks': [v + 1 for v in rank]  # ranks start from 1
                }
                results.append(row)
                if 'gt_index' in ann[idx]:
                    gt_id = ann[idx]['gt_index']
                    gt_rank = float(rank[gt_id])
                    rat1 = int(gt_rank < 1)
                    rat5 = int(gt_rank < 5)
                    rat10 = int(gt_rank < 10)
                    mean_ranks.append(float(gt_rank) + 1)
                    rat1s.append(float(rat1))
                    rat5s.append(float(rat5))
                    rat10s.append(float(rat10))

    mrrs = (1 / np.array(mean_ranks)).mean()
    mean_ranks = np.array(mean_ranks).mean()
    rat1s = np.array(rat1s).mean()
    rat5s = np.array(rat5s).mean()
    rat10s = np.array(rat10s).mean()
    stats = {
        'mrrs': f'{mrrs:.02f}',
        'mean_ranks': f'{mean_ranks:.02f}',
        'rat1': f'{rat1s:.02f}',
        'rat5': f'{rat5s:.02f}',
        'rat10': f'{rat10s:.02f}',
    }
    print(json.dumps(stats, indent=4))
    name = 'ref' if use_ref else 'mult'
    with open(f'../data/visdial/gen_{split}_{name}.json', 'w') as f:
        json.dump(results, f)

    print('done')
    return stats
