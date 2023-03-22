import os
import logging
import random
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch

from arguments import get_args
from infers.infer import main as infer_base
from infers.clip_infer import main as infer_clip
from infers.visdial import main as infer_visdial
from infers.merge import run_merge


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


args = get_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def run(args):
    if args.infer_force_imagefolder:
        args.infer_ann_path = None

    ann_path = str(args.infer_ann_path) if args.infer_ann_path is not None else ''
    args.use_visdial = 'visdial' in ann_path
    args.use_nocaps = 'nocaps' in ann_path
    args.use_goodnews = 'goodnews' in ann_path
    args.use_coco = 'coco' in ann_path
    args.use_style = 'style' in ann_path
    args.use_visualnews = 'visualnews' in ann_path
    args.use_image_folder = False
    if not (args.use_visdial or args.use_nocaps or args.use_goodnews or args.use_coco
            or args.use_style or args.use_visualnews):
        args.use_image_folder = True

    if args.use_visdial:
        log.info("running visdial inference")
        res = infer_visdial(args)
    elif args.infer_mode == 'base':
        log.info("running base inference")
        res = infer_base(args)
    elif args.infer_mode == 'clip':
        log.info("running clip score inference")
        res = infer_clip(args)
    return res


if len(args.infer_multi_checkpoint) > 0:
    run_merge(args, run)
elif args.infer_prompts is not None:
    prompts = args.infer_prompts.split(',')
    out_path = Path(args.infer_out_path)
    for prompt in tqdm(prompts, desc='prompts'):
        prompt = prompt.strip()
        for i in range(args.infer_sample_repeat):
            log.info(f"using prompt: ({prompt}), iter ({i + 1}/{args.infer_sample_repeat})")
            args.fixed_prompt = prompt
            name = prompt.replace(' ', '_')
            path = out_path.parent / f'{out_path.stem}_{name}_{i}_seed_{args.seed}.json'
            args.infer_out_path = str(path)
            run(args)
else:
    run(args)
