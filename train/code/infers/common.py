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
from data import ClipCocoCollator
from utils.utils import (
    get_first_sentence, remove_eot, get_first_dot,
    update_weight
)
from metric.lang_metrics import Eval
from .coco import CocoInferDataset
from .style import StyleInferDataset
from .nocaps import NocapsInferDataset
from .goodnews import GoodNewsInferDataset
from .visualnews import VisualNewsInferDataset
from .image_folder import ImageFolderInferDataset



logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_ann(args):
    if args.infer_ann_path is None:
        return None
    ann_path = Path(args.infer_ann_path)
    if args.use_nocaps:
        if ann_path.is_file():
            with open(ann_path) as f:
                ann = json.load(f)
            ann = ann['annotations']
            res = defaultdict(lambda: [])
            for row in ann:
                res[row['image_id']].append(row['caption'])
            ann = {str(k): v for k, v in res.items()}
            return ann
    elif args.use_goodnews:
        if ann_path.is_file():
            with open(ann_path) as f:
                ann = json.load(f)
            res = {}
            for row in ann:
                image_id = row['imgid']
                caption = row['sentences_full'][0]['raw'].strip()
                res[image_id] = [caption]
            return res
    elif args.use_visualnews:
        if ann_path.is_file():
            with open(ann_path) as f:
                ann = json.load(f)
            res = {}
            for row in ann:
                image_id = row['image_path']
                caption = row['caption']
                res[image_id] = [caption]
            return res
    elif args.use_style:
        if ann_path.is_file():
            with open(ann_path) as f:
                ann = json.load(f)
            res = {}
            for row in ann['annotations']:
                image_id = row['image_id']
                caption = {k.split('_')[1]: v for k, v in row.items()
                           if k.startswith('caption_')}
                res[int(image_id)] = caption
            return res
    else:
        if ann_path.is_file():
            with open(ann_path) as f:
                ann = json.load(f)
            ann = ann['images']
            # ann = {'/'.join(v['filepath'], v['filename']): [sent['raw'] for sent in v['sentences']]
            ann = {str(v['filename']): [sent['raw'] for sent in v['sentences']]
                for v in ann if v['split'] == args.infer_split}
            return ann
    return None


def load_model_args(args):
    if args.checkpoint is None:
        args.root_dir = '../data/log/temp'
        Path(args.root_dir).mkdir(exist_ok=True, parents=True)
    else:
        args.checkpoint = Path(args.checkpoint)
        assert args.checkpoint.is_file(), f"no checkpoint file: {args.checkpoint}"
        args.root_dir = str(args.checkpoint.parent.parent)
        args.checkpoint = str(args.checkpoint)
        args_path = Path(args.root_dir) / 'args.json'
        if args_path.is_file():
            with open(args_path) as f:
                hparams = json.load(f)
        else:
            args_path = Path(args.root_dir) / 'hparams.yaml'
            with open(args_path) as f:
                hparams = yaml.safe_load(f)
        for key in ['init_model', 'clip_model_type', 'use_caption', 'use_style_reward', 'use_transformer_mapper',
                    'prefix_length', 'clipcap_num_layers', 'use_ptuning_v2']:
            if key in hparams:
                setattr(args, key, hparams[key])
        # style model
        if 'transformer' in hparams:
            args.init_model = hparams['transformer']
        args.loaded_init_model = hparams.get('init_model', None)

    if args.infer_out_path is None:
        args.infer_out_path = str(Path(args.root_dir) / 'generations.json')

    return args


def load_model(args, device, finetune=False):
    log.info('loading model')
    def load_policy(args):
        policy = Policy(model_name=args.init_model, temperature=args.temperature, device=device,
                        clipcap_path=args.clipcap_path, fix_gpt=args.fix_gpt,
                        label_path=args.label_path,
                        prefix_length=args.prefix_length,
                        clipcap_num_layers=args.clipcap_num_layers,
                        use_transformer_mapper=args.use_transformer_mapper,
                        model_weight=args.init_model_weight, use_label_prefix=args.use_label_prefix)
        state = torch.load(args.checkpoint)
        policy_key = 'policy_model'
        if policy_key in state:
            policy.model.load_state_dict(state[policy_key])
        else:
            step = state['global_step']
            log.info(f'trained for {step} steps')
            weights = state['state_dict']
            key = 'policy.model.'
            if not any(k for k in weights.keys() if k.startswith(key)):
                key = 'model.model.'
            weights = {k[len(key):]: v for k, v in weights.items() if k.startswith(key)}
            # weights = {k: v for k, v in weights.items() if k.startswith('clip_project.')}
            weights = update_weight(policy, weights)
            policy.model.load_state_dict(weights, strict=False)
        return policy

    def load_style(args):
        model = AutoModelForCausalLM.from_pretrained(args.init_model)
        if args.checkpoint and Path(args.checkpoint).is_file():
            log.info("loading pretrained style generator")
            state = torch.load(args.checkpoint)
            if 'global_step' in state:
                step = state['global_step']
                log.info(f'trained for {step} steps')
            weights = state['state_dict']
            key = 'model.'
            weights = {k[len(key):]: v for k, v in weights.items() if k.startswith(key)}
            model.load_state_dict(weights)
            model = RefPolicy('gpt2', model=model, temperature=0.7, device=device)
        else:
            log.info("loading vanila gpt")
            model = Policy(model_name=args.init_model, temperature=args.temperature, device=device,
                        clipcap_path=args.clipcap_path, fix_gpt=args.fix_gpt,
                        label_path=args.label_path,
                        prefix_length=args.prefix_length,
                        clipcap_num_layers=args.clipcap_num_layers,
                        use_transformer_mapper=args.use_transformer_mapper,
                        model_weight=args.init_model_weight, use_label_prefix=args.use_label_prefix)
        return model

    if args.checkpoint is None:
        if Path(args.clipcap_path).is_file():
            log.info("loading vanila gpt with clipcap")
            model = Policy(model_name=args.init_model, temperature=args.temperature, device=device,
                        clipcap_path=args.clipcap_path, fix_gpt=args.fix_gpt,
                        label_path=args.label_path,
                        prefix_length=args.prefix_length,
                        clipcap_num_layers=args.clipcap_num_layers,
                        use_transformer_mapper=args.use_transformer_mapper,
                        model_weight=args.init_model_weight, use_label_prefix=args.use_label_prefix)
        else:
            log.info("loading base gpt")
            if finetune:
                model = Policy(model_name=args.init_model, temperature=args.temperature, device=device,
                             clipcap_path=args.clipcap_path, fix_gpt=args.fix_gpt,
                             prefix_length=args.prefix_length,
                             clipcap_num_layers=args.clipcap_num_layers,
                             label_path=args.label_path,
                             use_transformer_mapper=args.use_transformer_mapper,
                             use_ptuning_v2=args.use_ptuning_v2,
                             model_weight=args.init_model_weight, use_label_prefix=args.use_label_prefix)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.init_model)
                model = RefPolicy('gpt2', model=model, temperature=0.7, device=device)
    elif args.loaded_init_model is not None:
        model = load_policy(args)
    else:
        model = load_style(args)
    model = model.to(device)
    return model


def load_model_with_args(args, device):
    args = load_model_args(args)
    model = load_model(args, device)
    return args, model


def prepare(args):
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args.label_path = Path(args.label_path)
    args.infer_dir = Path(args.infer_dir)

    args, model = load_model_with_args(args, device)

    labels = None
    args.use_labels = args.label_path != Path('None')
    if args.use_labels:
        with open(args.label_path) as f:
            labels = json.load(f)
        labels = list(labels.keys())
    anns = load_ann(args)

    log.info('loading data')
    if args.use_nocaps:
        InferDataset = NocapsInferDataset
    elif args.use_goodnews:
        InferDataset = GoodNewsInferDataset
    elif args.use_visualnews:
        InferDataset = VisualNewsInferDataset
    elif args.use_style:
        InferDataset = StyleInferDataset
    elif args.use_coco:
        InferDataset = CocoInferDataset
    else:
        InferDataset = ImageFolderInferDataset
    data = InferDataset(args.ref_model, args.clip_model_type, args.use_caption,
                        fixed_prompt=args.fixed_prompt,
                        label_path=args.label_path,
                        use_label_prefix=args.use_label_prefix,
                        infer_dir=args.infer_dir)
    collator = ClipCocoCollator(data.tokenizer)

    def get_batch(ids):
        nonlocal labels
        if labels is None:
            batch = [[v, ''] for v in ids]
        else:
            batch = list(itertools.product(ids, labels))
        meta = batch
        batch = [data.getitem(*v) for v in batch]
        domains = None
        if 'domain' in batch[0]:
            domains = [v['domain'] for v in batch]
        batch = collator(batch)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch['domains'] = domains
        return meta, batch

    if args.infer_do_sample:
        print("sampling")
    if args.infer_num is not None and args.infer_num < args.batch_size:
        args.batch_size = args.infer_num

    if args.use_visualnews:
        image_ids = list(data.captions_raw.keys())  # random sampled test set
    elif anns is None:
        assert args.infer_dir.is_dir(), f"no inference directory: {args.infer_dir}"
        image_ids = sorted([*[v.name for v in args.infer_dir.glob('*.jpg')],
                            *[v.name for v in args.infer_dir.glob('*.png')]])
    else:
        image_ids = sorted(list(anns.keys()))
    image_ids = image_ids[:args.infer_num]
    log.info(f'number of images to infer: {len(image_ids)}')
    data.anns = anns
    return args, model, data.tokenizer, data, image_ids, get_batch, device


def main(args, sanity_check=False, infer_f=None):
    print("inference")
    assert infer_f is not None, "please provide an inference function"
    args, model, tokenizer, data, image_ids, get_batch, device = prepare(args)
    chunks = list(get_chunks(image_ids, args.batch_size))
    clip_model, clip_preprocess = clip.load(args.clip_model_type, device=device, jit=False)
    if hasattr(args, 'clip_infer') and args.clip_infer:
        model = model.model

    def get_clip_score(texts, image_features):
        text = clip.tokenize(texts, truncate=True).to(device)
        text_features = clip_model.encode_text(text)
        image_features = image_features.clone().to(device).to(text_features.dtype)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        cossim = torch.einsum('bc,bc->b', image_features, text_features)
        cossim = cossim.detach().cpu().numpy()
        clip_score = np.maximum(cossim, 0) * 2.5
        return clip_score.tolist()

    out = defaultdict(lambda: {})
    clip_scores = defaultdict(lambda: {})
    domains = defaultdict(lambda: [])
    # add_prompt = not args.use_labels
    for j, chunk in tqdm(enumerate(chunks), total=len(chunks)):
        meta, batch = get_batch(chunk)
        if batch['features'] is None:
            images = [clip_preprocess(image) for image in batch['images']]
            images = torch.stack(images, dim=0).to(device)
            batch['features'] = clip_model.encode_image(images)

        if 'domains' in batch and batch['domains'] is not None:
            for iid, domain in zip(batch['image_ids'], batch['domains']):
                domains[domain].append(iid)

        if args.use_visualnews:
            batch['raw_prefixes'] = batch['prefixes']

        if args.infer_do_sample > 0:
            storage = defaultdict(list)
            for i in range(args.infer_do_sample):
                query, generations = infer_f(args, model, tokenizer,
                                             clip_model, clip_preprocess,
                                             batch,
                                             infer_clip_scale=args.infer_clip_scale,
                                             infer_ce_scale=args.infer_ce_scale,
                                             sample=True)
                with torch.no_grad():
                    clip_score = get_clip_score(generations, batch['features'])
                storage['g'].append(generations)
                storage['q'].append(query)
                storage['c'].append(clip_score)
            scores = np.array(storage['c'])
            ids = scores.argmax(0)[None, :].repeat(scores.shape[0], axis=0)
            generations = np.take_along_axis(np.array(storage['g']), ids, axis=0)[0]
            query = np.take_along_axis(np.array(storage['q']), ids, axis=0)[0]
            clip_score = np.take_along_axis(np.array(storage['c']), ids, axis=0)[0]
        else:
            query, generations = infer_f(args, model, tokenizer,
                                         clip_model, clip_preprocess,
                                         batch,
                                         infer_clip_scale=args.infer_clip_scale,
                                         infer_ce_scale=args.infer_ce_scale,
                                         sample=False)

            with torch.no_grad():
                clip_score = get_clip_score(generations, batch['features'])

        '''
        if add_prompt:
            # generations = [f'{q.strip()} {g.strip()}' for q, g in zip(query, generations)]
            generations = [f'A {g.strip()}' for q, g in zip(query, generations)]
        '''
        if sanity_check:
            generations = [data.anns[v[0]][0] for v in batch['image_ids']]
            torch.cuda.empty_cache()
            with torch.no_grad():
                clip_score = get_clip_score(generations, batch['features'])

        for i, ((image_id, label), generation, q, cs) in enumerate(zip(meta, generations, query, clip_score)):
            out[image_id][label] = generation
            clip_scores[image_id][label] = cs
            if j < 3 and i < 1:
                tqdm.write(f"{image_id}, ({q}), {generation}")

        if args.infer_num is not None and args.infer_num <= (j + 1) * args.batch_size:
            break

    out = dict(out)
    if args.use_nocaps:
        metrics = Eval(use_spice=True)
    else:
        metrics = Eval(use_spice=False)  # spice takes long

    def get_score(xcs, xo, eval_label='caption'):
        stats = {}
        if args.use_labels:
            clip_score = [v[eval_label] for k, v in xcs.items()]
        else:
            clip_score = [v[''] for k, v in xcs.items()]
        clip_score = np.array(clip_score).mean()
        stats['clip_score'] = clip_score
        if data.anns is not None:
            if args.use_labels:
                hypos = {k: v[eval_label] for k, v in xo.items()}
            else:
                hypos = {k: v[''] for k, v in xo.items()}
            tgts = data.anns
            keys = list(hypos.keys())
            hypos = [hypos[k] for k in keys]
            tgts = [tgts[k] for k in keys]
            if args.use_nocaps or args.use_coco:
                hypos = [get_first_dot(remove_eot(v)) for v in hypos]
            stats = {**stats, **metrics(hypos, tgts), 'num': len(tgts)}
        return stats

    if args.use_nocaps:
        stats = {}
        for k, domain in domains.items():
            print(f'Domain: {k}')
            domain = set(domain)
            xcs = {k: v for k, v in clip_scores.items() if k in domain}
            xo = {k: v for k, v in out.items() if k in domain}
            stat = get_score(xcs, xo)
            stats[k] = stat

        stat = get_score(clip_scores, out)
        stats['overall'] = stat

        means = defaultdict(lambda: [])
        for domain, stat in stats.items():
            if domain == 'overall':
                continue
            for k, v in stat.items():
                means[k].append(v)

        means = {k: sum(v) if k == 'num' else np.array(v).mean() for k, v in means.items()}
        stats['style'] = means
    elif args.use_visualnews:
        stats = {}
        for k, domain in domains.items():
            print(f'Domain: {k}')
            domain = set(domain)
            xcs = {k: v for k, v in clip_scores.items() if k in domain}
            xo = {k: v for k, v in out.items() if k in domain}
            label = k.split('_')[0]
            stat = get_score(xcs, xo, eval_label=label)
            stats[k] = stat
        stat = get_score(clip_scores, out, eval_label='news')
        stats['overall'] = stat
    elif args.use_style:
        hypos = defaultdict(lambda: {})
        hypo_cs = defaultdict(lambda: {})
        tgts = defaultdict(lambda: {})
        for image_id, row in out.items():
            for label, sent in row.items():
                hypos[label][image_id] = sent
                hypo_cs[label][image_id] = clip_scores[image_id][label]
                if label in data.anns[image_id]:
                    tgts[label][image_id] = data.anns[image_id][label]

        stats = {}

        def cut(x):
            return tokenizer.decode(tokenizer.encode(x)[:args.response_length])

        for label, hypo in hypos.items():
            if label in tgts:
                hypo_c = hypo_cs[label]
                tgt = tgts[label]
                keys = list(tgt.keys())
                hypo = [hypo[k] for k in keys]
                tgt = [tgt[k] if isinstance(tgt[k], list) else cut(tgt[k]) for k in keys]
                cs = np.array(list(hypo_c.values())).mean()
                stat = metrics(hypo, tgt)
                stats[label] = {**stat, 'clip_score': cs, 'length': len(hypo)}
    else:
        stats = get_score(clip_scores, out)

    print(json.dumps(stats, indent=4))

    res = {'generations': out, 'stats': stats, 'fixed_prompt': args.fixed_prompt}
    with open(args.infer_out_path, 'w') as f:
        json.dump(res, f, indent=4)
    return stats


if __name__ == "__main__":
    args = get_args()
    main(args, sanity_check=False)
