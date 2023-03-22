import os
import json
from pathlib import Path
from itertools import chain

from tqdm import tqdm
import numpy as np
import torch

from policy import Policy
from reward import Reward
from torch.utils.data import DataLoader
from main import ClipCocoDataset, ClipCocoCollator
from utils.utils import load_jsonl, ensure_dir, reduce_sum

path = '../data/log/ppo/generations/large'
model = 'gpt2-large'
clip_model_type = 'ViT-B/32'
use_caption = False
batch_size = 4
rate_limit = 15
num_samples = 25
checkpoint_path = '../data/log/ppo/03-18-2022_15:17:19/model/ckp_7000.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ensure_dir(path)

policy = Policy(model_name=model, temperature=1.0, device=device)
reward_model = Reward(gain=1, bias=0,
                clip_model_type=clip_model_type, device=device)
# ref_policy = Policy(model_name='gpt2-xl', temperature=1.0, device=device)
prompt_collator = ClipCocoCollator(tokenizer=policy.tokenizer)

if checkpoint_path is not None:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    policy.model.load_state_dict(checkpoint['policy_model'])

print('model initialization done!')
val_dataset = ClipCocoDataset(model_name=model, split='val',
                              clip_model_type=clip_model_type, use_caption=use_caption)
dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=prompt_collator)


def expand(tensor, num_repeat):
    if len(tensor.shape) == 2:
        return torch.reshape(tensor[:, None].expand(-1, num_repeat, -1), [batch_size * num_repeat, -1])
    elif len(tensor.shape) == 3:
        return torch.reshape(tensor[:, None].expand(-1, num_repeat, -1, -1), [batch_size * num_repeat, -1, -1])
    else:
        raise Exception('tensor shape not supported')


def expand_list(li, num_repeat):
    li = [[v for _ in range(num_repeat)] for v in li]
    li = list(chain(*li))
    return li


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def distinctness(responses, num_sample):
    generations_batch = list(chunks(responses, num_sample))
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for generations in tqdm(generations_batch, total=len(generations_batch), desc='Evaluating diversity'):
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)

    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


image_ids, responses, scores = [], [], []
for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
    image_id, input_ids, attention_mask, features = batch

    features = expand(features, num_samples)
    outputs = policy.sample(input_ids=expand(input_ids, num_samples),
                            attention_mask=expand(attention_mask, num_samples),
                            features=features,
                            top_p=0.9)
    prompt, response = outputs['query/text'], outputs['response/text']
    score = reward_model.get_reward(features, response, '')
    image_ids = [*image_ids, *expand_list(image_id, num_samples)]
    responses = [*responses, *response]
    scores = [*scores, *score]
    if i >= 9:
        break

# print(f"average perplexity = {mean(perplexities):+.2f}")
data = [{'image_id': x, 'response': y, 'score': z} for x, y, z in zip(image_ids, responses, scores)]
data = list(chunks(list(zip(image_ids, responses, scores)), num_samples))
data = [{'image_id': row[0][0], 'responses': [{'response': v[1], 'score': v[2]} for v in row]}
        for row in data]

scores = torch.Tensor(scores).reshape(-1, num_samples)
max_cos_sim = scores.max(dim=1).values.float().mean()
cos_sim_prob = (scores > 0.5).float().max(dim=1).values.mean()

print(f'average maximum cos_sim = {max_cos_sim:.3f}')
print(f'average cos_sim probability = {cos_sim_prob:.3f}')

dist1, dist2, dist3 = distinctness(responses, num_samples)
print(f'dist-1={dist1:.3f}, dist-2={dist2:.3f}, dist-3={dist3:.3f}')

# write output results
with open(f'{path}/eval_results.txt', 'w') as fo:
    fo.write(f'average maximum cos_sim = {max_cos_sim:.3f}\n')
    fo.write(f'average cos_sim probability = {cos_sim_prob:.3f}\n')
    fo.write(f'dist-1={dist1:.3f}, dist-2={dist2:.3f}, dist-3={dist3:.3f}\n')
    for line in data[:10]:
        fo.write(f'{json.dumps(line)}\n')
