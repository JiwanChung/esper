import os
import math
import random
import logging
import json
import pickle
from pathlib import Path
from itertools import chain
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback
from transformers import AutoTokenizer

from infers.common import load_model_args, load_model
from finetune import Trainer, main
from utils.utils import reduce_sum


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class VisDialValDataset(torch.utils.data.Dataset):
    def __init__(self, model_name: str, clip_model_type: str, infer_num: Optional[int] = None):
        self.clip_model_type = clip_model_type
        self.clip_model_name = self.clip_model_type.replace('/', '_')
        self.num_val = infer_num

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.load_data()
        self.get_indices()

    def load_ann(self, split, num_dialogs=None):
        with open(f'../data/visdial/visdial_1.0_{split}.json', 'r') as f:
            data = json.load(f)
        data = data['data']
        res = {}
        images = set()
        corrects = {}
        for did, row in enumerate(data['dialogs']):
            idx = row['image_id']
            images.add(idx)
            for i, turn in enumerate(row['dialog']):
                question = turn['question']
                if 'answer_options' not in turn:
                    continue
                gt_index = turn['gt_index']
                cands = turn['answer_options']
                question = data['questions'][question]
                cands = [data['answers'][cand] for cand in cands]
                filename = f'VisualDialog_{split}2018_{idx:012d}.jpg'

                irid = f'{idx}_{i}'
                for j, cand in enumerate(cands):
                    answer_true = j == gt_index
                    if answer_true:
                        corrects[irid] = j
                    out = {'id': f"{idx}_{i}", 'image_id': f'{idx}', 'round_id': f'{i}',
                           'question': question, 'candidate': cand, 'answer_id': j,
                           'answer_true': answer_true,
                           'gt_id': gt_index,
                           'filename': filename}
                    full_id = f'{idx}_{i}_{j}'
                    res[full_id] = out
            if num_dialogs is not None and did >= num_dialogs:
                break
        self.anns = res
        self.images = list(images)
        self.corrects = corrects

    def __len__(self):
        return len(self.ids)

    def get_indices(self):
        self.ids = list(self.anns)

    def load_data(self):
        self.load_ann('val', self.num_val)
        data_dir = Path('../data/visdial/cache/clipcap')
        with open(data_dir / f'{self.clip_model_name}_val_vision.pkl', 'rb') as f:
            all_data = pickle.load(f)
        prefixes = all_data["clip_embedding"].reshape(-1, 512)
        captions_raw = all_data["captions"]
        captions_raw = {v['id']: {**v, 'candidates': v['candidates'].split('<eos>')}
                                  for v in captions_raw}
        self.prefixes = prefixes
        self.captions_raw = captions_raw

    def get_feature(self, image_id: str, round_id, answer_id):
        full_id = f'{image_id}_{round_id}_{answer_id}'
        dt = self.anns[full_id]
        idx = f'{image_id}_{round_id}'
        temp = self.captions_raw[idx]
        dt['clip_embedding'] = temp['clip_embedding']
        # dt['candidate'] = dt['candidates'][answer_id]
        return dt, self.prefixes[dt['clip_embedding']]

    def getitem(self, image_id: str, round_id, answer_id):
        dt, feature = self.get_feature(image_id, round_id, answer_id)
        if dt is None:
            return None
        candidate = dt['candidate']
        # candidate = f'B: {v.strip()}'
        question = dt['question']
        res = {
            'image_id': int(dt['image_id']),
            'round_id': int(dt['round_id']),
            'answer_id': int(dt['answer_id']),
            'question': question,
            'candidate': candidate,
            'feature': feature,
            'correct': dt['answer_true'],
            'gt_id': dt['gt_id']
        }
        return res

    def __getitem__(self, idx):
        full_id = self.ids[idx]
        image_id, round_id, answer_id = full_id.split('_')
        return self.getitem(image_id, int(round_id), int(answer_id))


class VisDialTrainDataset(VisDialValDataset):
    def get_indices(self):
        self.ids = list(self.images)

    def load_data(self):
        self.load_ann('train')
        data_dir = Path('../data/coco/cache/clipcap')

        self.prefixes = {}
        self.captions_raw = {}
        for split in ['train', 'val', 'test']:
            with open(data_dir / f'{self.clip_model_name}_{split}_vision.pkl', 'rb') as f:
                all_data = pickle.load(f)
            self.prefixes[split] = all_data["clip_embedding"]
            captions_raw = all_data["captions"]
            self.captions_raw[split] = {v['filename']: v for v in captions_raw}

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        round_id = random.randrange(0, 10)
        return [self.getitem(image_id, round_id, answer_id)
                for answer_id in range(100)]

    def get_feature(self, image_id: str, round_id, answer_id):
        full_id = f'{image_id}_{round_id}_{answer_id}'
        dt = self.anns[full_id]
        iid = int(dt['image_id'])
        filename_tr = f'COCO_train2014_{iid:012d}.jpg'
        filename_val = f'COCO_val2014_{iid:012d}.jpg'
        flag = False
        dt_split = 'train'
        for split in ['train', 'val', 'test']:
            if filename_tr in self.captions_raw[split]:
                temp = self.captions_raw[split][filename_tr]
                dt_split = split
                flag = True
                break
            elif filename_val in self.captions_raw[split]:
                temp = self.captions_raw[split][filename_val]
                dt_split = split
                flag = True
                break
        if not flag:
            raise Exception(f'image file {iid} not found in the feature cache')
        dt['clip_embedding'] = temp['clip_embedding']
        return dt, self.prefixes[dt_split][dt['clip_embedding']]


class VisDialBinaryTrainDataset(VisDialTrainDataset):
    def __getitem__(self, idx):
        image_id = self.ids[idx]
        round_id = random.randrange(0, 10)
        irid = f'{image_id}_{round_id}'
        gt_id = self.corrects[irid]
        cands = list(range(100))
        cands.pop(gt_id)
        false_id = random.choice(cands)
        return [self.getitem(image_id, round_id, gt_id),
                self.getitem(image_id, round_id, false_id)]


class VisDialCollator:
    def __init__(self, tokenizer, use_yes, use_ll):
        self.tokenizer = tokenizer
        self.use_yes = use_yes
        self.use_ll = use_ll
        if self.use_yes:
            self.ans_idx_true = self.tokenizer.encode(' Yes')[0]
            self.ans_idx_false = self.tokenizer.encode(' No')[0]
        else:
            self.ans_idx_true = self.tokenizer.encode(' True')[0]
            self.ans_idx_false = self.tokenizer.encode(' False')[0]

    def __call__(self, sequences):
        sequences = [seq for seq in sequences if seq is not None]
        image_ids = [sequence['image_id'] for sequence in sequences]
        round_ids = [sequence['round_id'] for sequence in sequences]
        answer_ids = [sequence['answer_id'] for sequence in sequences]
        gt_ids = [sequence['gt_id'] for sequence in sequences]
        correct = [sequence['correct'] for sequence in sequences]
        features = [sequence['feature'] for sequence in sequences]
        features = torch.stack(features, dim=0)
        questions = [sequence['question'] for sequence in sequences]

        candidate = [sequence['candidate'] for sequence in sequences]

        # queries = ['dialogue: A:' for _ in questions]
        if self.use_ll:
            queries = [f'dialogue: A: {q}\n' for q,c in zip(questions, candidate)]
        else:
            queries = [f'dialogue: A:' for q,c in zip(questions, candidate)]

        encodings_dict = self.tokenizer(queries, return_tensors="pt", padding=True)
        query_input_ids = encodings_dict['input_ids']
        query_mask = encodings_dict['attention_mask']

        # candidate = [f'{q}\nB:{c.strip()}\nTrue' for q,c in zip(questions, candidate)]
        if self.use_ll:
            candidate = [f"B:{c.strip()}" for q, c in zip(questions, candidate)]
        elif self.use_yes:
            candidate = [f"{q}\nB:{c.strip().replace('Yes', 'yes')}\nA: Yes" for q,c in zip(questions, candidate)]
        else:
            candidate = [f"{q}\nB:{c.strip().replace('True', 'true')}\nB: True" for q,c in zip(questions, candidate)]
        encodings_dict = self.tokenizer(candidate, return_tensors="pt", padding=True)
        response_input_ids = encodings_dict['input_ids']
        response_mask = encodings_dict['attention_mask']
        if self.use_ll:
            answer_pos = None
        else:
            answer_mask = response_input_ids == self.ans_idx_true
            answer_pos = answer_mask.nonzero()[:, 1]

        correct = torch.tensor(correct).bool()

        batch = {
            'image_ids': image_ids,
            'round_ids': round_ids,
            'answer_ids': answer_ids,
            'query_input_ids': query_input_ids,
            'query_mask': query_mask,
            'response_input_ids': response_input_ids,
            'response_mask': response_mask,
            'features': features,
            'answer_pos': answer_pos,
            'gt_ids': gt_ids,
            'corrects': correct
        }
        return batch


class VisDialTrainCollator(VisDialCollator):
    def __call__(self, batch):
        batch = list(chain(*batch))
        return super().__call__(batch)



def get_visdial_metrics(ranks):
    stats = {}
    for at in [1, 5, 10]:
        val = (ranks <= at).astype(float).mean()
        stats[f'R@{at}'] = val
    stats['mean_rank'] = ranks.astype(float).mean()
    stats['MRR'] = (1 / ranks).astype(float).mean()
    return stats


class VisDialCallback(Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.val_data = {
            'hypo': defaultdict(lambda: {}),
            'gt': {}
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        data = pl_module.val_data
        results = []
        data_rank = []
        gt_vals  = []
        mean_vals  = []
        max_vals  = []
        for key in data['hypo'].keys():
            hypo = data['hypo'][key]
            gt = data['gt'][key]

            ids, val = zip(*hypo.items())
            ids = torch.tensor(ids).long()
            val = torch.tensor(val).float()
            # val = F.softmax(val, dim=-1)  # logits to probs
            gt_val = val[gt]
            mean_val = val.mean()
            max_val = val.max()
            order = val.argsort(descending=True)
            ranks = order.argsort()
            rank = ranks[gt].item()
            data_rank.append(rank + 1)  # ranks start from 1
            image_id, round_id = key.split('_')
            row = {
                'image_id': int(image_id),
                'round_id': int(round_id) + 1,  # rounds start from 1
                'ranks': (ranks + 1).cpu().numpy().tolist()  # ranks start from 1
            }
            results.append(row)
            gt_vals.append(gt_val)
            mean_vals.append(mean_val)
            max_vals.append(max_val)

        batch_size = len(data_rank)
        metrics = get_visdial_metrics(np.array(data_rank))
        metrics = {**metrics, 'gt_val': np.array(gt_vals).mean(),
                   'mean_val': np.array(mean_vals).mean(),
                   'max_val': np.array(max_vals).mean()}
        print('------')
        print('epoch metrics')
        for k, v in metrics.items():
            print(f"  {k} = {v:+.2f}")
            pl_module.log(f'Evaluation/{k}', v, on_epoch=True, batch_size=batch_size)
        print('------')

        log_dir = pl_module.hparams.save_dir
        coco_dir = Path(log_dir) / 'vals'
        coco_dir.mkdir(exist_ok=True)

        with open(coco_dir / f'step_{pl_module.global_step}.json', 'w') as f:
            json.dump(results, f, indent=4)


class VisDialTrainer(Trainer):
    def setup(self, stage: str = None):
        device = self.device
        log.info(f'Loading model')
        self.model = load_model(self.hparams, device, finetune=True)
        gpt = self.model.model.gpt if hasattr(self.model.model, 'gpt') else self.model.model
        if self.hparams.fix_gpt:
            log.info(f'fixing gpt weights')
            for param in gpt.parameters():
                param.requires_grad_(False)
        else:
            log.info(f'unfreezing gpt weights')
            for param in gpt.parameters():
                param.requires_grad_(True)
        # unfreezing classifier
        for param in gpt.lm_head.parameters():
            param.requires_grad_(True)

        log.info(f'Loading data')
        TrainDataset = VisDialTrainDataset
        if self.hparams.visdial_binary_loss:
            TrainDataset = VisDialBinaryTrainDataset
        train_dataset = TrainDataset(model_name=self.hparams.init_model,
                                     clip_model_type=self.hparams.clip_model_type)

        prompt_tokenizer = train_dataset.tokenizer
        val_collator = VisDialCollator(tokenizer=prompt_tokenizer,
                                       use_ll=self.hparams.visdial_likelihood_loss,
                                       use_yes=self.hparams.visdial_use_yes)
        train_collator = VisDialTrainCollator(tokenizer=prompt_tokenizer,
                                       use_ll=self.hparams.visdial_likelihood_loss,
                                       use_yes=self.hparams.visdial_use_yes)
        self.collator = val_collator
        batch_size = 1
        if self.hparams.visdial_binary_loss:
            batch_size = 50
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      num_workers=self.hparams.num_workers,
                                      shuffle=True, drop_last=True, collate_fn=train_collator)
        log.info(f'Load train set with {len(train_dataset)} examples')

        test_dataset = VisDialValDataset(model_name=self.hparams.init_model,
                                         clip_model_type=self.hparams.clip_model_type,
                                         infer_num=self.hparams.infer_num)
        test_dataloader = DataLoader(test_dataset, batch_size=200,
                                    num_workers=self.hparams.num_workers,
                                    shuffle=False, collate_fn=val_collator)
        log.info(f'Load val set with {len(test_dataset)} examples')
        if self.hparams.num_epochs is not None:
            log.info(f'Total Epochs {self.hparams.num_epochs}')
            self.hparams.total_steps = math.ceil(self.hparams.num_epochs * len(train_dataloader) / self.hparams.grad_acc)
            log.info(f'Total Steps {self.hparams.total_steps}')
        self.loaders = {'train': train_dataloader, 'test': test_dataloader}

    def run(self, batch):
        forward_inputs = {'query_input_ids': batch['query_input_ids'],
                          'query_mask': batch['query_mask'],
                          'response_input_ids': batch['response_input_ids'],
                          'response_mask': batch['response_mask'],
                          'features': batch['features']}
        outputs = self.model.forward_pass(**forward_inputs, invalidate_eos=False,
                                           device=self.device)
        ids = batch['answer_pos']  # answer T/F position indices
        if self.hparams.visdial_likelihood_loss:
            logprobs = outputs['response/log_prob']  # BLV
            mask = batch['response_mask']
            logprobs = reduce_sum(logprobs, mask, -1)
            ans_logit = logprobs
        else:
            '''
            logits = outputs['response/logits']  # BLV
            logits = F.log_softmax(logits, dim=-1)
            true = logits[:, :, self.collator.ans_idx_true]
            false = logits[:, :, self.collator.ans_idx_false]
            logit = true - false  # log ratio
            '''
            if self.hparams.visdial_normalized_loss:
                logit = outputs['response/pos_logit']
            else:
                logit = outputs['response/log_prob']
            ans_logit = logit.gather(1, ids[:, None]).squeeze(1)
        return outputs, ans_logit

    def get_loss(self, batch, ans_logit):
        if self.hparams.visdial_binary_loss:
            ans_logit = ans_logit.reshape(-1, 2)
            logprob = F.log_softmax(ans_logit, dim=-1)  # B 2
            loss = -logprob[:, 0].mean()  # true sample first
            acc = (logprob[:, 0] > logprob[:, 1]).float().mean()
        else:
            ids = torch.tensor(batch['gt_ids']).to(ans_logit.device).long()
            ids = ids.reshape(-1, 100)
            cut_ids = ids[:, 0]
            assert (ids == cut_ids[:, None]).all()
            ids = cut_ids
            ans_logit = ans_logit.reshape(-1, 100)
            if self.hparams.visdial_likelihood_loss:
                logprob = ans_logit
            else:
                logprob = F.log_softmax(ans_logit, dim=-1)  # B 100
            loss = logprob.gather(1, ids[:, None])
            loss = -loss.mean()
            acc = (logprob.argmax(-1) == ids).float().mean()
        return loss, acc

    def training_step(self, batch):
        sched = self.lr_schedulers()
        if not isinstance(sched, list):
            sched = [sched]
        for _sched in sched:
            _sched.step()
        opt = self.optimizers()
        if not isinstance(opt, list):
            opt = [opt]
        for _opt in opt:
            _opt.zero_grad()

        outputs, ans_logit = self.run(batch)
        loss, acc = self.get_loss(batch, ans_logit)
        batch_size = ans_logit.shape[0]
        self.log(f'Train/Loss', loss.item(), on_step=True, batch_size=batch_size)
        self.log(f'Train/Acc', acc.item(), on_step=True, batch_size=batch_size)

        self.manual_backward(loss)
        for _opt in opt:
            _opt.step()
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs, ans_logit = self.run(batch)
            batch_size = ans_logit.shape[0]

            val = ans_logit.cpu().numpy().tolist()
            for iid, rid, aid, v, gt in zip(batch['image_ids'], batch['round_ids'],
                                        batch['answer_ids'], val, batch['gt_ids']):
                idx = f'{iid}_{rid}'
                self.val_data['hypo'][idx][aid] = v
                self.val_data['gt'][idx] = gt


if __name__ == "__main__":
    main('Evaluation/MRR', VisDialTrainer, VisDialCallback)
