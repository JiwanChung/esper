import os
import math
import json
import logging
from pathlib import Path

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from pytorch_lightning.utilities.apply_func import apply_to_collection

from utils.utils import reduce_sum, chunk_list
from utils.constants import NEGATIVE_INF


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


def _move_float_tensors_to_half(batch):
    batch = apply_to_collection(batch, (torch.FloatTensor, torch.cuda.FloatTensor), function=lambda x: x.half())
    return batch


class Buffer:
    def __init__(self, args, train_dataloader, collator,
                 text_model=None, pair_model=None, device=None):
        self.args = args
        self.selected_prior = 20
        self.rep_thres = args.replay_repetition_threshold  # 2
        self.rep_ngram = args.replay_repetition_ngram  # 2
        self.text_thres = args.replay_text_threshold  # -60
        self.pair_thres = args.replay_pair_threshold  # 0.4
        self.age_thres = args.replay_age_threshold  # None
        self.min_replay_size = args.min_buffer_size  # 1024
        self.new_gen_ratio = args.replay_generated_sample_ratio
        self.batch_size = args.batch_size
        self.response_length = args.response_length
        self.max_buffer_size = args.max_buffer_size
        self.pad_id = text_model.tokenizer.pad_token_id

        dataset = train_dataloader.dataset
        if args.replay_init_path is not None:
            path = Path(args.replay_init_path)
            if path.is_file():
                log.info(f"loading replay_init from: {path}")
                with open(path) as f:
                    init = json.load(f)
                rows = []
                for i, (filename, text) in enumerate(init.items()):
                    idx = dataset.ids_map[filename][0]
                    row = dataset.getitem(idx)
                    if text.startswith('A '):
                        text = text[2:].strip().capitalize()
                    if args.use_label_prefix:
                        row['caption'] = 'caption:'
                        row['text'] = f' {text}'
                        row['prefix'] = ''
                    else:
                        row['caption'] = 'A'
                        row['text'] = f' {text}'
                        row['prefix'] = row['caption']
                    rows.append(row)

                chunks = list(chunk_list(rows, args.minibatch_size))
                for batch in tqdm(chunks, total=len(chunks), desc="loading replay inits"):
                    texts = [v['text'] for v in batch]
                    query_texts = [v['caption'] for v in batch]
                    batch = [{k: v2 for k, v2 in v.items() if k != 'text'} for v in batch]
                    batch = collator(batch)
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                    with torch.no_grad():
                        response_texts, inputs = build_continuations(text_model, batch, texts,
                                                       query_texts, device, self.response_length)

                    response = text_model.tokenizer(response_texts, return_tensors="pt", padding=True,
                                                    truncation=True, max_length=self.response_length).to(device)
                    response_texts = text_model.tokenizer.batch_decode(response['input_ids'])
                    inputs = text_model.tokenizer(inputs, return_tensors="pt", padding=True,
                                                    truncation=True, max_length=self.response_length).to(device)
                    rollouts = {'query/input_ids': batch['input_ids'].long(),
                                'query/text': query_texts,
                                'query/mask': batch['attention_mask'].long(),
                                'response/input_ids': response['input_ids'],
                                'response/text': response_texts,
                                'response/mask': response['attention_mask'],
                                'inputs/mask': inputs['attention_mask'],
                    }

                    self.insert(rollouts, batch, text_model, pair_model, device, buffer_name='offline',
                                add_age=False)
                    del rollouts
                    del batch

        # self._remove('offline', self.max_buffer_size)

    def __len__(self):
        if hasattr(self, 'online'):
            return len(self.online)
        else:
            return 0

    def get_size(self, name):
        if hasattr(self, name):
            return len(getattr(self, name))
        else:
            return 0

    def get_ce_target(self, log_probs, mask):
        prob = min(self.args.ce_target_prob, 1)
        log_prob = math.log(prob)
        for b in range(mask.shape[0]):
            idx = mask[b].sum()
            log_probs[b, :idx] = log_prob
        return log_probs

    def calc_scores(self, rollouts, batch, text_model, pair_model, device='cuda', **kwargs):
        features = batch['features']
        with torch.no_grad():
            forward_inputs = {
                'query_input_ids': rollouts['query/input_ids'],
                'query_mask': rollouts['query/mask'],
                'response_input_ids': rollouts['response/input_ids'],
                'response_mask': rollouts['response/mask'],
            }
            out = text_model.forward_pass(**forward_inputs, invalidate_eos=True, device=device)
            out.pop('response/entropy')
            if self.args.ce_target_prob > 0 and 'inputs/mask' in rollouts:
                out['response/log_prob'] = self.get_ce_target(out['response/log_prob'], rollouts['inputs/mask'])
            rollouts = {**rollouts, **out,
                        'response/ref_log_prob': out['response/log_prob'],
                        'response/ref_eos_prob': out['response/eos_prob']}

            logprobs = rollouts['response/log_prob']
            response_masks = rollouts['response/mask']
            responses = rollouts['response/text']
            queries = rollouts['query/text']
            texts = [v1 + v2 for v1, v2 in zip(queries, responses)]

            text_scores = reduce_sum(logprobs, response_masks, axis=1)
            pair_scores = pair_model._get_reward(features, texts, device=device)
            normed_pair = pair_model.normalize(pair_scores)
            pair_scores = [max(v, 0) * 2.5 for v in pair_scores]

            rep_scores = [count_rep(v, self.rep_ngram) for v in responses]

        text_scores = text_scores.cpu().numpy().tolist()
        rollouts = wrap(rollouts)
        batch = wrap(batch)
        return [{'rollouts': a, 'batch': b,
                 'text_score': c, 'pair_score': d, 'rep_score': e,
                 'normed_pair': f}
                for a, b, c, d, e, f
                in zip(rollouts, batch,
                       text_scores, pair_scores, rep_scores,
                       normed_pair)]

    def insert(self, *args, add_age=True, buffer_name='online', **kwargs):
        scores = self.calc_scores(*args, **kwargs)
        scores = pd.DataFrame(scores)
        scores['age'] = 1
        scores['selected'] = 0
        scores['sort_score'] = self.get_sort_score(scores)

        if hasattr(self, buffer_name):
            if add_age:
                getattr(self, buffer_name)['age'] += 1
            setattr(self, buffer_name, pd.concat([getattr(self, buffer_name), scores]))
        else:
            setattr(self, buffer_name, scores)
        if buffer_name != 'offline':
            self.new_gen = scores

    def remove(self, max_replay_size=2048):
        for buffer_name in ['online']:
            self._remove(buffer_name, max_replay_size)

    def _remove(self, buffer_name, max_replay_size=2048):
        if hasattr(self, buffer_name):
            replay = getattr(self, buffer_name)
            replay = replay.loc[replay.rep_score < self.rep_thres]
            replay = replay.loc[replay.text_score > self.text_thres]
            replay = replay.loc[replay.pair_score > self.pair_thres]

            if self.age_thres:
                old = replay.loc[replay.age > self.age_thres].index
                old = old[:4]  # do not remove much at onces
                if len(replay) - len(old) >= self.min_replay_size:
                    replay = replay.drop(index=old)

            if len(replay) >= max_replay_size:
                replay = replay.nlargest(max_replay_size, 'sort_score')

                setattr(self, buffer_name, replay)

    def get_sort_score(self, scores):
        x = scores['text_score']
        score1 = scores['text_score'].clip(self.text_thres, -10)
        score1 = np.log(score1 - self.text_thres + 5)  # 1.6 ~ 4.5
        score2 = scores['pair_score']
        score3 = 1 / (scores['rep_score'] + 1)
        return score1 * score2 * score3

    def postprocess_samples(self, rollouts, batch, scores, names, device):
        rollouts = unwrap(rollouts, self.pad_id, device)
        batch = unwrap(batch, self.pad_id, device)
        labels = batch['labels']
        if labels is not None:
            labels = torch.tensor(labels).long().to(device)
        scores = torch.tensor(scores).to(device)
        batch['labels'] = labels
        rollouts['replay_names'] = names
        return rollouts, batch, scores

    def merge_samples(self, samples):
        rollouts = []
        batch = []
        scores = []
        names = []
        for name, sample in samples:
            if sample is not None:
                s_rollouts = sample['rollouts'].tolist()
                s_batch = sample['batch'].tolist()
                s_scores = sample['normed_pair'].tolist()
                s_names = [name for _ in range(len(s_scores))]
                rollouts.extend(s_rollouts)
                batch.extend(s_batch)
                scores.extend(s_scores)
                names.extend(s_names)
        return rollouts, batch, scores, names

    def get_new_gen_samples(self, size):
        new_gen = self.new_gen
        if len(new_gen) > size:
            sample_size = min(math.ceil(size * 2), len(new_gen))
            new_gen = new_gen.nlargest(sample_size, 'sort_score')
            new_gen = new_gen.sample(size)
        return new_gen

    def get_replay_samples(self, size, buffer_name='offline'):
        buffer_ = getattr(self, buffer_name)
        if self.args.replay_deterministic:
            weight = 1 / (buffer_['selected'] + self.selected_prior)  # counter deprivation
            weight = weight * self.get_sort_score(buffer_)
            buffer_['weight'] = weight
            return buffer_.nlargest(size, 'weight')
        else:
            weight = self.get_sort_score(buffer_)
            return buffer_.sample(size, weights=weight)

    def sample(self, sample_size, offline_ratio=0.2, device='cuda'):
        size_offline = min(sample_size, math.ceil(sample_size * offline_ratio))
        size_new_gen = min(sample_size - size_offline, math.ceil(sample_size * self.new_gen_ratio))
        size_online = max(0, sample_size - size_new_gen - size_offline)
        online_samples = None
        if size_online > 0:
            online_samples = self.get_replay_samples(size_online, 'online')
            ids = online_samples.index
            self.online.loc[ids, 'selected'] += 1
        offline_samples = None
        if size_offline > 0:
            offline_samples = self.get_replay_samples(size_offline, 'offline')
            ids = offline_samples.index
            self.offline.loc[ids, 'selected'] += 1
        new_gen_samples = self.get_new_gen_samples(size_new_gen)
        rollouts, batch, scores, names = self.merge_samples([('new_gen', new_gen_samples),
                                                      ('online', online_samples),
                                                      ('offline', offline_samples)])

        return self.postprocess_samples(rollouts, batch, scores, names, device)

    def get_rep_score(self, buffer_name):
        if hasattr(self, buffer_name):
            scores = getattr(self, buffer_name)['rep_score'].tolist()
            scores = torch.tensor(scores).float().mean().item()
            return scores
        return 0

    def get_clip_score(self, buffer_name):
        if hasattr(self, buffer_name):
            scores = getattr(self, buffer_name)['pair_score'].tolist()
            scores = torch.tensor(scores).mean().item()
            return scores
        return 0

    def get_ref_entropy(self, buffer_name):
        if hasattr(self, buffer_name):
            scores = getattr(self, buffer_name)['text_score'].tolist()
            scores = -torch.tensor(scores).mean().item()
            return scores
        return 0

    def get_kl(self, buffer_name, model, device):
        if hasattr(self, buffer_name):
            buffer_ = getattr(self, buffer_name)
            with torch.no_grad():
                rollouts = buffer_['rollouts'].tolist()
                batch = buffer_['batch'].tolist()
                rollouts = unwrap(rollouts, self.pad_id, device)
                batch = unwrap(batch, self.pad_id, device)
                forward_inputs = {
                    'query_input_ids': rollouts['query/input_ids'],
                    'query_mask': rollouts['query/mask'],
                    'response_input_ids': rollouts['response/input_ids'],
                    'response_mask': rollouts['response/mask'],
                    'features': batch['features']
                }
                out = model.forward_pass(**forward_inputs, invalidate_eos=True, device=device)
                logprob = out['response/log_prob']
                ref_logprob = rollouts['response/ref_log_prob']
                mask = rollouts['response/mask']
                x = reduce_sum(logprob - ref_logprob, mask, axis=1).cpu()
            x = torch.mean(x).item()
            return x
        return 0

    def get_ages(self, buffer_name):
        if hasattr(self, buffer_name):
            return getattr(self, buffer_name)['age'].mean()
        return 0

    def get_num_selecteds(self, buffer_name):
        if hasattr(self, buffer_name):
            return getattr(self, buffer_name)['selected'].mean()
        return 0

    def get_offline_ratio(self, step):
        step = max(0, step - self.args.decay_offline_ratio_after)
        step = math.sqrt(step + 1) if step > 0 else 0
        return max(self.args.min_offline_ratio,
                   self.args.max_offline_ratio +
                   min(0, -self.args.offline_ratio_growth * step))


def wrap(x):
    if isinstance(x, (list, tuple)):
        x = [v.detach().cpu().numpy() if torch.is_tensor(v) else v for v in x]
        x = [[v[k] if v is not None else v for v in x] for k in range(len(x[0]))]
        return x
    if isinstance(x, dict):
        x = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in x.items()}
        pivot = list(x.keys())[0]
        x = [{k: v[i] if v is not None else v for k, v in x.items()} for i in range(len(x[pivot]))]
        return x
    else:
        raise Exception(f'invalid col_type: {type(x)}')


def _to_tensor(v, pad_id=0):
    batch_size = len(v)
    length = max([len(row) for row in v])
    storage = torch.full((batch_size, length), pad_id)
    for i, row in enumerate(v):
        row = torch.from_numpy(row)
        if i == 0:
            storage = storage.to(row.dtype)
        storage[i, :len(row)] = row
    return storage


def to_tensor(v, name, pad_id=0):
    if name.endswith('prob'):
        pad_id = NEGATIVE_INF
    elif 'ids' not in name:
        pad_id = 0
    return _to_tensor(v, pad_id)


def unwrap(x, pad_id=0, device='cuda'):
    batch_name_map = {1: 'input_ids', 2: 'attention_mask', 3: 'features'}
    if isinstance(x[0], (list, tuple)):
        flags = [isinstance(v, np.ndarray) for v in x[0]]
        x = [[v[k] for v in x] for k in range(len(x[0]))]
        x = [to_tensor(v, batch_name_map[i], pad_id=pad_id).to(device) if flag else v
             for i, (v, flag) in enumerate(zip(x, flags))]
        x = [v if v[0] is not None else None for v in x]
        return x
    elif isinstance(x[0], dict):
        flags = {k: isinstance(v, np.ndarray) for k, v in x[0].items()}
        x = {k: [v[k] for v in x] for k in x[0].keys()}
        x = {k: to_tensor(v, k, pad_id=pad_id).to(device) if flags[k] else v
             for k, v in x.items()}
        x = {k: v if v[0] is not None else None for k, v in x.items()}
        return x
    else:
        raise Exception(f'invalid col_type: {type(x[0])}')


def count_rep(x, n=2):
    x = np.array(x.split(' '))
    orig_length = len(x)
    xs = []
    for i in range(n):
        xs.append(np.roll(x, -i, 0))
    x = np.stack(xs, 1)
    uniques = np.unique(x, axis=0)
    return orig_length - len(uniques)


def build_continuations(model, batch, texts, captions, device, length):
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    # input_texts = [v1 + v2 for v1, v2 in zip(captions, texts)]
    input_texts = texts
    inputs = model.tokenizer(input_texts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=length).to(device)

    rollouts = model.sample(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
        invalidate_eos=True,
        max_len=length * 2,
        device=device)
    res = rollouts['response/text']
    res = [v1 + v2 for v1, v2 in zip(input_texts, res)]
    return res, input_texts


if __name__ == "__main__":
    text = "I don't know. I don't know."
    rep = count_rep(text)
    print(rep)
    text = "I don't know."
    rep = count_rep(text)
    print(rep)
