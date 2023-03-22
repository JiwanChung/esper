import torch

from .common import main as run
from utils.utils import get_first_sentence


def base_infer(args, model, tokenizer,
               clip, clip_preprocess,
               batch, sample=False, temperature=0.7,
               **kwargs):
    with torch.no_grad():
        rollouts = model.sample(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                features=batch['features'], labels=None,
                                temperature=temperature,
                                max_len=args.response_length, sample=sample,
                                no_repeat_ngram_size=args.infer_no_repeat_size,
                                invalidate_eos=not args.eos_ok)
        generations = rollouts['response/text']
        # generations = [get_first_sentence(v) for v in generations]
        query = rollouts['query/text']
        del rollouts
    torch.cuda.empty_cache()
    return query, generations


def main(args):
    return run(args, infer_f=base_infer)


if __name__ == "__main__":
    args = get_args()
    main(args)
