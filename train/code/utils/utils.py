import json
from pathlib import Path
from typing import TypeVar, Iterable, List, Union, Any
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
import collections
from utils.constants import NEGATIVE_INF


T = TypeVar('T')


def reduce_sum(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask)
    return torch.sum(value * mask, axis)


def reduce_mean(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask) / torch.sum(mask)
    return reduce_sum(value, mask, axis) / torch.sum(mask, axis)


def reduce_std(value, mask):
    return torch.sqrt(reduce_mean(torch.square(value), mask) - torch.square(reduce_mean(value, mask)))


def logits_to_entropy(logits):
    distribution = torch.distributions.Categorical(logits=logits)
    return distribution.entropy()


def mask_pad(value, mask):
    return value * mask + NEGATIVE_INF * (1 - mask)


def clamp(value, min_value, max_value):
    return torch.max(torch.min(value, max_value), min_value)


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
    return q


def whiten(values, masks, shift_mean=True):
    mean, var = reduce_mean(values, masks), reduce_std(values, masks)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def flatten_dict(nested, sep='.'):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f'Loading cache from {file}'):
                yield json.loads(line)


def get_first_sentence(txt, min_len=5):
    eos = '<|endoftext|>'
    eos_idx = txt.find(eos)
    if eos_idx > 0:
        txt = txt[eos_idx:]
    txt = txt.replace('\n', ' ')
    sents = txt.split('. ')
    if len(sents[0]) >= min_len:
        sent = f'{sents[0].strip()}.'
    else:
        sent = txt
    return sent


def remove_eot(txt):
    eos = '<|endoftext|>'
    eos_idx = txt.find(eos)
    if eos_idx > 0:
        txt = txt[:eos_idx]
    return txt


def get_longest(txts):
    lens = [len(v) for v in txts]
    idx = np.array(lens).argmax()
    return txts[idx]


def get_jsd(net_1_logits, net_2_logits):
    net_1_probs =  F.softmax(net_1_logits, dim=1)
    net_2_probs = F.softmax(net_2_logits, dim=1)

    m = 0.5 * (net_1_probs + net_1_probs)
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), m, reduction="none")
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), m, reduction="none")

    return (0.5 * loss)


def get_kl(net_1_logits, net_2_logits, normalized=False):
    if not normalized:
        log_probs_1 = F.log_softmax(net_1_logits, dim=1)
        log_probs_2 = F.log_softmax(net_2_logits, dim=1)
    else:
        log_probs_1 = net_1_logits
        log_probs_2 = net_2_logits

    loss = F.kl_div(log_probs_1, log_probs_2, reduction="none",
                    log_target=True)

    return (0.5 * loss)


def get_mean_kl(net_1_logits, net_2_logits):
    net_1_probs = F.softmax(net_1_logits, dim=1).detach()
    net_2_probs = F.softmax(net_2_logits, dim=1).detach()

    loss_1 = F.kl_div(F.log_softmax(net_1_logits, dim=1), net_2_probs, reduction="none")
    loss_2 = F.kl_div(F.log_softmax(net_2_logits, dim=1), net_1_probs, reduction="none")
    loss = (loss_1 + loss_2) / 2

    return (0.5 * loss)


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_first_dot(sent):
    if '.' in sent:
        sent = sent[:sent.find('.')]
    return sent


def get_chunks(li, size):
    return [li[i:i + size] for i in range(0, len(li), size)]


def update_weight(model, weight, fname=''):
    key = f'{fname}gpt.transformer.wte.weight'
    if key in weight:
        w = model.model.gpt.transformer.wte.weight.data.clone()
        _w = weight[key]
        w = w.detach().to(_w.device).to(_w.dtype)
        if len(w) >= len(_w):
            w[:len(_w), :] = _w
            weight[key] = w
        else:
            weight[key] = _w[:len(w)]

    key = f'{fname}gpt.lm_head.weight'
    if key in weight:
        w = model.model.gpt.lm_head.weight.data.clone()
        _w = weight[key]
        w = w.detach().to(_w.device).to(_w.dtype)
        if len(w) >= len(_w):
            w[:len(_w), :] = _w
            weight[key] = w
        else:
            weight[key] = _w[:len(w)]
    return weight
