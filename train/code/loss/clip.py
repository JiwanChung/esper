import math
from itertools import chain

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils.utils import get_kl, reduce_mean, reduce_sum


def get_steps(tokenizer, ids, probs, pos_ids, device, k=512, j=32):
    storage = []
    xids = []
    for i, pos in enumerate(pos_ids):
        pos = pos.item()
        prob = probs[pos]
        topk = prob.topk(k=k)
        tokens = topk.indices.squeeze()
        values = topk.values.squeeze()
        if j < k:
            rand_w = values
            rand_ids = rand_w.multinomial(num_samples=j, replacement=False)
            tokens = tokens[rand_ids]
        if i == 0:
            x = tokens[:, None]
        else:
            x = torch.cat((ids[:pos][None, :].repeat(j, 1), tokens[:, None]), dim=1)
        xids.append(tokens)
        x = tokenizer.batch_decode(x)
        storage.append(x)
    xids = torch.stack(xids, dim=0)
    return storage, xids


def get_clip_loss(reward_model, tokenizer, features, query, input_ids, logits, masks,
                  device, rand_pos=-1, num_samples=32, tau=1):
    B, L = input_ids.shape[:2]
    V = len(tokenizer)
    if rand_pos > 0:
        pos_ids = torch.randint(0, 10, (B, rand_pos)).to(device)
    else:
        pos_ids = torch.arange(0, L)[None, :].repeat(B, 1).to(device)
    #  get target
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        xidss = []
        stepss = []
        for i in range(B):
            steps, xids = get_steps(tokenizer, input_ids[i], probs[i], pos_ids[i], device,
                                    j=num_samples)
            steps = [[f'{query[i]} {res.strip()}' for res in step] for step in steps]
            steps = list(chain(*steps))
            stepss.extend(steps)
            xidss.append(xids)
        feat = features[i][None, :].repeat(len(stepss), 1)
        rewards = reward_model._get_reward(feat, stepss, device=device)
        log_potential = rewards / tau
        pss = torch.from_numpy(log_potential).to(device)
        pss = pss.reshape(B, xids.shape[0], -1)
        xidss = torch.stack(xidss, dim=0) # B P J
        tgt = pss
        '''
        tgt = torch.zeros_like(pss)[:, :, 0][:, :, None].repeat(1, 1, V)
        tgt = tgt.scatter(dim=2, src=pss, index=xidss)  # B L V
        '''

    masks = masks.gather(dim=1, index=pos_ids)
    logits = logits.gather(dim=1, index=pos_ids[:, :, None].repeat(1, 1, V))  # B P V
    logits = logits.gather(dim=2, index=xidss)  # B P J

    # calc ce loss
    B, L, V = logits.shape
    tgt = tgt.reshape(-1, V)
    logits = logits.reshape(-1, V)
    log_probs = F.log_softmax(logits, dim=1)
    loss = get_kl(log_probs, tgt, normalized=True)
    clip_mask = (tgt >= math.log(0.1)).to(masks.dtype)
    loss = (loss * clip_mask).mean(-1)
    loss = loss.reshape(B, -1)
    loss = torch.mean(reduce_mean(loss, masks, axis=1))
    return loss
