import os
import logging
from pathlib import Path

import yaml
import numpy as np
import torch
from transformers import AutoModelForCausalLM


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


def get_transformer_module(Module, default_name, **kwargs):
    if default_name == 'EleutherAI/gpt-j-6B':
        kwargs = {**kwargs, **dict(revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)}
    model = Module.from_pretrained(default_name, **kwargs)
    return model


def load_weights(self, Module, path, name, default_name, prev_name=None, **kwargs):
    hparams = None
    if path is not None and Path(path).is_dir():
        path = Path(path)
        weight_path = path / 'checkpoints' / 'best.ckpt'
        if weight_path.is_file():
            print(f"loading pretrained weights from: {path}")

            with open(path / 'hparams.yaml', 'r') as f:
                hparams = yaml.safe_load(f)

            weight = torch.load(weight_path)['state_dict']
            if prev_name is not None:
                weight = {k.replace(prev_name, name): v for k, v in weight.items()}

            if 'init_model' in hparams:
                transformer = hparams['init_model']
            else:
                transformer = hparams['transformer']
            model = Module.from_pretrained(transformer, **kwargs)
            setattr(self, name, model)
            if 'init_model' not in hparams:
                self.load_state_dict(weight)
        else:
            raise Exception(f"no best.ckpt found in: {weight_path}")
    else:
        assert isinstance(default_name, str), f'invalid default transformer name: {default_name}'
        model = get_transformer_module(Module, default_name, **kwargs)
        setattr(self, name, model)
    return hparams


def check_ckpt_loadable(args, ckpt):
    if args.use_deepspeed:
        param_file = ckpt.parent.parent / 'hparams.yaml'
        with open(param_file, 'r') as f:
            old_params = yaml.safe_load(f)
        if old_params['num_gpus'] != args.num_gpus:
            print(f"found a ckpt, but the number of gpus is different")
            print(f"current: {args.num_gpus}, ckpt: {old_params['num_gpus']}")
            print("the number of gpus has to match for deepspeed to work")
            print("starting training without resuming from checkpoint")
            ckpt = None
    return ckpt


def find_last_checkpoint(args):
    root = args.save_dir

    # load ckpt from arguments if possible
    if isinstance(args.checkpoint, str):
        ckpt = Path(args.checkpoint)
        if ckpt.is_dir():
            return check_ckpt_loadable(args, ckpt)

    # load from previous trainings
    save_dir = Path(root) / 'default'
    new_save_dir = Path(root) / 'lightning_logs'
    if save_dir.is_dir():
        # dicts are already ordered in recent python versions
        ckpts = {int(p.parent.parent.stem.split('_')[1]): p for p in save_dir.rglob('version_*/checkpoints/last.ckpt')}
        if len(ckpts) > 0:
            keys = np.array(list(ckpts.keys()))
            key = keys.argsort()[-1]
            key = keys[key]
            ckpt = ckpts[key]
            return check_ckpt_loadable(args, ckpt)
    if new_save_dir.is_dir():
        ckpts = {int(p.parent.parent.stem.split('_')[1]): p for p in new_save_dir.rglob('version_*/checkpoints/last.ckpt')}
        if len(ckpts) > 0:
            keys = np.array(list(ckpts.keys()))
            key = keys.argsort()[-1]
            key = keys[key]
            ckpt = ckpts[key]
            return check_ckpt_loadable(args, ckpt)
    return None


def download_weights(args):
    log.info('downloading weights before multiprocessing')
    models = set([args.ref_model, args.init_model, args.init_value_model])
    for model in list(models):
        downloaded = get_transformer_module(AutoModelForCausalLM, model)
        del downloaded
    try:
        import clip
        clip_model, preprocess = clip.load(args.clip_model_type, device='cpu', jit=False)
        del clip_model
    except:
        import clip_jax
        image_fn, text_fn, jax_params, jax_preprocess = clip_jax.load('ViT-B/32', "cpu")
        del jax_params
    torch.cuda.empty_cache()
    return None


def load_finetuned(args, model):
    path = args.init_model_weight
    if path is not None and Path(path).is_dir():
        path = Path(path)
        weight_path = path / 'checkpoints' / 'best.ckpt'
        if weight_path.is_file():
            print(f"loading pretrained weights from: {path}")
            weight = torch.load(weight_path)['state_dict']
            weight = {k[len('model.'):]: v for k, v in weight.items()}
            model.load_state_dict(weight)
        else:
            raise Exception(f"no best.ckpt found in: {weight_path}")
    return model
