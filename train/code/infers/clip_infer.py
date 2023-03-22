import os
import json
import pickle
import random
import logging

from tqdm import tqdm
import torch
import clip

from .common import main as run
from .ZeroCLIP import CLIPTextGenerator
from arguments import get_args
from utils.utils import get_first_sentence, remove_eot


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


def clip_infer(args, model, tokenizer,
               clip_model, clip_preprocess,
               batch, sample=False, temperature=0.7,
               beam_size=5,
               infer_clip_scale=0.5,
               infer_ce_scale=0.5):
    gen = CLIPTextGenerator(
        model, tokenizer, clip_model, clip_preprocess,
        target_seq_length=args.response_length,
        clip_scale=infer_clip_scale,
        ce_scale=infer_ce_scale,
        device=batch['features'].device
    )
    query = []
    generations = []
    for i, feat in tqdm(enumerate(batch['features']), desc='in_batch_iter'):
        cond_text = tokenizer.decode(batch['input_ids'][i])
        cond_text = remove_eot(cond_text)
        feat = feat.unsqueeze(0)
        captions = gen.run(feat, cond_text, beam_size=beam_size, verbose=False)
        encoded_captions = [gen.clip.encode_text(clip.tokenize(c, truncate=True).to(gen.device)) for c in captions]
        encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
        best_clip_idx = (torch.cat(encoded_captions) @ feat.t()).squeeze().argmax().item()
        caption = captions[best_clip_idx]
        tqdm.write(f'({cond_text}) {caption}')
        query.append(cond_text)
        generations.append(caption)
    torch.cuda.empty_cache()
    # generations = [get_first_sentence(v) for v in generations]
    return query, generations


def main(args):
    args.clip_infer = True
    return run(args, infer_f=clip_infer)


if __name__ == "__main__":
    args = get_args()
    main(args)
