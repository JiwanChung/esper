import os
import json
import argparse
import logging
from pathlib import Path

import yaml
import torch
from utils.constants import (
    GAIN, BIAS, STYLE_GAIN, STYLE_BIAS,
    STYLE_ACC_GAIN, STYLE_ACC_BIAS
)


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument(
        '--config', type=str, default='', help='config file path')
    parser.add_argument(
        '--debug', action='store_true', default=False, help='enable debug mode')
    parser.add_argument(
        '--use_deepspeed', action='store_true', default=False, help='deepspeed model parallel')
    parser.add_argument(
        '--disable_auto_resume', action='store_true', default=False, help='disable auto resume from checkpoints')
    parser.add_argument(
        '--fast_dev_run', action='store_true', default=False, help='fast dev run for debugging')
    parser.add_argument(
        '--num_gpus', type=int, default=None, help='number of gpus to use')
    parser.add_argument(
        '--num_workers', type=int, default=8, help='number of workers for dataloading')
    parser.add_argument(
        '--use_ref_gen_until', type=int, default=None, help='use ref policy for generating samples until step N to stabilize training')
    parser.add_argument(
        '--use_only_kl_until', type=int, default=None, help='do not optimize reward until step N')
    parser.add_argument(
        '--sampling_entropy_threshold', type=int, default=None, help='If entropy is bigger than this, we discard all samples and sample again. Ignored if None')
    parser.add_argument(
        '--alt_loss', action='store_true', default=False, help='alternative ppo loss')
    parser.add_argument(
        '--disable_dropout', action='store_true', default=False, help='disable dropout on train')

    # replay
    parser.add_argument(
        '--use_replay_buffer', action='store_true', default=False, help='replay_buffer')
    parser.add_argument(
        '--replay_init_path', type=str, default=None, help='pre-generated samples for replay buffer init')
    parser.add_argument(
        '--ce_target_prob', type=float, default=-1, help='target value for offline buffer data. Not used if <= 0')
    parser.add_argument(
        '--min_buffer_size', type=int, default=1024, help='min buffer size for replay')
    parser.add_argument(
        '--max_buffer_size', type=int, default=2048, help='max buffer size for replay')
    parser.add_argument(
        '--buffer_growth', type=float, default=1, help='buffer size = growth * num_step')
    parser.add_argument(
        '--replay_repetition_threshold', type=float, default=2, help='max repetition to allow')
    parser.add_argument(
        '--replay_repetition_ngram', type=int, default=2, help='size of ngrams to count repetitions with')
    parser.add_argument(
        '--replay_text_threshold', type=float, default=-60, help='min text logprob to allow')
    parser.add_argument(
        '--replay_pair_threshold', type=float, default=0.4, help='min clip score to allow')
    parser.add_argument(
        '--replay_age_threshold', type=int, default=None, help='max age to allow')
    parser.add_argument(
        '--replay_generated_sample_ratio', type=float, default=0.25, help='generated_sample_ratio')
    parser.add_argument(
        '--replay_deterministic', action='store_true', help='get topk samples from buffer rather than sampling')
    parser.add_argument(
        '--decay_offline_ratio_after', type=int, default=0, help='decay offline ratio after step')
    parser.add_argument(
        '--min_offline_ratio', type=float, default=0.05, help='max ratio of samples to get from the offline buffer (starting point)')
    parser.add_argument(
        '--max_offline_ratio', type=float, default=0.5, help='max ratio of samples to get from the offline buffer (starting point)')
    parser.add_argument(
        '--offline_ratio_growth', type=float, default=1/27.6, help='growth coefficient for ratio of samples to get from the offline buffer')

    # dataset
    parser.add_argument(
        '--output-dir', type=str, default='../outputs')
    parser.add_argument(
        '--use_open_images', action='store_true', help='use additional open images')
    parser.add_argument(
        '--use_only_open_images', action='store_true', help='use open images without COCO')
    parser.add_argument(
        '--use_goodnews_images', action='store_true', help='use additional goodnews train set images')
    parser.add_argument(
        '--debug_replicate_images', action='store_true', help='replicate images for debugging')
    '''
    parser.add_argument(
        '--dataset-train', type=str, default='data/toxicity/train.jsonl',
        help='JSONL file containing train prompts. Each row must contain a prompt at `row["prompt"]["text"]`.')
    parser.add_argument(
        '--dataset-val', type=str, default='data/toxicity/val.jsonl',
        help='JSONL file containing dev prompts. Each row must contain a prompt at `row["prompt"]["text"]`.')
    '''

    # policy
    parser.add_argument(
        '--init-model', type=str, default='gpt2', help='language model used for policy.')
    parser.add_argument(
        '--init-value-model', type=str, default=None, help='language model used for value.')
    parser.add_argument(
        '--ref-model', type=str, default='gpt2', help='language model used for reference policy.')
    parser.add_argument(
        '--response-length', type=int, default=20, help='number of tokens to generate for each prompt.')
    parser.add_argument(
        '--temperature', type=float, default=0.7, help='temperature for sampling policy.')
    parser.add_argument(
        '--eos_ok', action='store_true', default=False, help='do not set eos probability to negative infinity')
    parser.add_argument(
        '--no_repeat_ngram_size', type=int, default=0, help='no repeat ngram size for episode sampling. not used in eval.')

    # ppo
    parser.add_argument(
        '--total-episodes', type=int, default=10000000, help='total number of episodes')
    parser.add_argument(
        '--num_epochs', type=int, default=None, help='override total_episodes')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='batch size')
    parser.add_argument(
        '--warmup_steps', type=int, default=0, help='number of warmup steps for lr scheduler')
    parser.add_argument(
        '--nminibatches', type=int, default=16, help='number of ppo minibatch per batch')
    parser.add_argument(
        '--noptepochs', type=int, default=4, help='number of ppo epochs reusing rollouts')
    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument(
        '--grad_acc', type=int, default=1, help='gradient accumulation')
    parser.add_argument(
        '--pg_coef', type=float, default=1.0, help='policy gradient loss coefficient')
    parser.add_argument(
        '--vf_coef', type=float, default=1.0, help='value loss coefficient')
    parser.add_argument(
        '--clip_coef', type=float, default=1.0, help='clip loss coefficient')
    parser.add_argument(
        '--cliprange', type=float, default=.2, help='clip parameter for policy gradient')
    parser.add_argument(
        '--cliprange_value', type=float, default=.2, help='clip parameter for value function')
    parser.add_argument(
        '--gamma', type=float, default=1.0, help='discount factor for rewards')
    parser.add_argument(
        '--lam', type=float, default=0.95, help='lambda parameter for generalized advantage estimation')
    parser.add_argument(
        '--whiten_rewards', action='store_false', default=True, help='whether to normalize reward in each minibatch')
    parser.add_argument(
        '--clip_grad', action='store_true', default=False, help='whether to clip gradient')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='maximum norm of gradients ')
    parser.add_argument(
        '--optimizer', type=str, default='AdamW', help='optimizer name (only works in lightning)')

    # reward
    parser.add_argument(
        '--ref_entropy_mask', type=float, default=None, help='ref entropy as score reward masks')
    parser.add_argument(
        '--kl_ref_entropy_mask', type=float, default=None, help='ref entropy as kl reward masks')
    parser.add_argument(
        '--do_not_use_goal_before', type=int, default=None, help='do not optimize for goal (e.g. pair or style) rewards before K iters')
    parser.add_argument(
        '--do_not_use_rewards_before', type=int, default=None, help='do not optimize rewards in policy net before K iters')
    parser.add_argument(
        '--kl_coef', type=float, default=0.15, help='coefficient for KL term in reward')
    parser.add_argument(
        '--kl_as_constraints', type=float, default=None, help='kl constraint threshold')
    parser.add_argument(
        '--supervise_kl_coef', type=float, default=1, help='kl supervision loss coefficient')
    parser.add_argument(
        '--supervise_kl', type=float, default=None, help='kl constraint threshold')
    parser.add_argument(
        '--supervise_l2', type=float, default=None, help='minimize l2 distance btw encoder output vs. generation embedding')
    parser.add_argument(
        '--supervise_ce', action='store_true', help='use ce loss instead of kl')
    parser.add_argument(
        '--use_clip_loss', action='store_true', help='use clip loss')
    parser.add_argument(
        '--use_entropy_loss', action='store_true', help='use entropy loss')
    parser.add_argument(
        '--entropy_threshold', type=float, default=70, help='entropy is ok below this')
    parser.add_argument(
        '--entropy_coef', type=float, default=0.2, help='scale of the entropy loss')
    parser.add_argument(
        '--clip_loss_rand_pos', type=int, default=2, help='number of random positions to calc clip loss with.')
    parser.add_argument(
        '--clip_loss_num_samples', type=int, default=32, help='number of random vocab to calc clip loss with.')
    parser.add_argument(
        '--entropy_reward_threshold', type=float, default=None, help='ref_policy entropy as rewards')
    parser.add_argument(
        '--entropy_reward_bias', type=float, default=0, help='entropy reward bias')
    parser.add_argument(
        '--entropy_reward_coef', type=float, default=1, help='entropy reward coef')
    parser.add_argument(
        '--repetition_penalty', type=float, default=None, help='repetition penalty rewards')
    parser.add_argument(
        '--repetition_reward_bias', type=float, default=0, help='repetition reward bias')
    parser.add_argument(
        '--use_gpt_tok_for_rp', action='store_true', default=False, help='use gpt tokenizer for repetition counting')
    parser.add_argument(
        '--kl_type', type=str, default='kl', help='kl computation type')
    parser.add_argument(
        '--adaptive_kl', action='store_true', default=False, help='whether to use adaptive KL controller')
    parser.add_argument(
        '--do_not_use_kl_rewards', action='store_true', default=False, help='to not use kl')
    parser.add_argument(
        '--target', type=float, default=6.0, help='target value in adaptive KL controller')
    parser.add_argument(
        '--horizon', type=float, default=10000, help='horizon value in adaptive KL controller')
    parser.add_argument(
        '--gain', type=float, default=GAIN, help='normalization factor for reward')
    parser.add_argument(
        '--bias', type=float, default=BIAS, help='normalization factor for reward')
    parser.add_argument(
        '--style_gain', type=float, default=STYLE_GAIN, help='normalization factor for style reward')
    parser.add_argument(
        '--style_bias', type=float, default=STYLE_BIAS, help='normalization factor for style reward')
    parser.add_argument(
        '--use_binary_visdial', action='store_true', default=False, help='higher reward for True/ lower reward for False')
    parser.add_argument(
        '--continual_supervision_coef', type=float, default=0, help='supervision_coef')
    parser.add_argument(
        '--supervision_tgt', type=str, default=None, help='supervision target file path')

    # clip
    parser.add_argument(
        '--prefix_length', type=int, default=10, help='prefix length for the visual mapper')
    parser.add_argument(
        '--clipcap_num_layers', type=int, default=1, help='num_layers for the visual mapper')
    parser.add_argument(
        '--pair_reward_exp', action='store_true', default=False, help='exponential scaling for clip rewards')
    parser.add_argument(
        '--clip_model_type', type=str, default='ViT-B/32', help='clip backbone type')
    parser.add_argument(
        '--clip_sent_split', type=str, default='none', help='sentence split method for clip eval')
    parser.add_argument(
        '--use_caption', action='store_true', default=False, help='use caption data as prefix')
    parser.add_argument(
        '--fix_gpt', action='store_true', default=False, help='fix gpt weights')
    parser.add_argument(
        '--use_ptuning_v2', action='store_true', default=False, help='use prefix tuning (ptuning v2)')
    parser.add_argument(
        '--use_transformer_mapper', action='store_true', default=False, help='use transformer mapper instead of mlp')
    parser.add_argument(
        '--clipcap_path', type=str, default='', help='pretrained clipcap weights')
    parser.add_argument(
        '--use_label_prefix', action='store_true', default=False, help='label as prefixes')
    parser.add_argument(
        '--init_model_weight', type=str, default='None', help='initial model pretrained weight path')
    parser.add_argument(
        '--init_value_model_weight', type=str, default='INIT', help='initial value model pretrained weight path')
    parser.add_argument(
        '--ref_model_weight', type=str, default='None', help='reference model pretrained weight path')
    parser.add_argument(
        '--unfix_value_model', action='store_true', help='unfix_value_model_only')
    parser.add_argument(
        '--use_resized_tokenizer', action='store_true', help='resized tokenizer for clipcap-coco')

    # style
    parser.add_argument(
        '--style_cache_path', type=str, default='None', help='pretrained style classifier weights')
    parser.add_argument(
        '--label_path', type=str, default='None', help='style label info file path')
    parser.add_argument(
        '--style_acc_as_rewards', action='store_true', default=False, help='use the accuracy as rewards instead of the negative CE loss')
    parser.add_argument(
        '--sent_init_path', type=str, default='None', help='file for sentence init tokens')

    # pair
    parser.add_argument(
        '--do_not_use_pair_reward', action='store_true', default=False, help='do not use pair reward')
    parser.add_argument(
        '--use_coco_eval', action='store_true', default=False, help='evaluate coco zero-shot captioning')
    parser.add_argument(
        '--fixed_prompt', type=str, default='', help='fixed prompt')
    parser.add_argument(
        '--eval_label', type=str, default='caption', help='label prompt for evaluation')

    # generation
    parser.add_argument(
        '--num-samples', type=int, default=25, help='number of samples to generate for each prompt.')
    parser.add_argument(
        '--top-p', type=float, default=1.0, help='hyperparameter for nucleus sampling')

    # other
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='step interval to print out logs')
    parser.add_argument(
        '--save-interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval-interval', type=int, default=500, help='step interval to do evaluation')
    parser.add_argument(
        '--cuda-deterministic', action='store_false', default=True,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--only_download', action='store_true', default=False, help='for saving cache')
    parser.add_argument(
        '--use_cpu', action='store_true', default=False, help='for saving cache')


    # inference
    parser.add_argument(
        '--checkpoint', type=str, default=None, help='checkpoint file path')
    parser.add_argument(
        '--infer_dir', type=str, default='None', help='directory with images to infer')
    parser.add_argument(
        '--infer_out_path', type=str, default=None, help='output file path')
    parser.add_argument(
        '--infer_ann_path', type=str, default=None, help='directory with the annotations file')
    parser.add_argument(
        '--infer_split', type=str, default='val', help='inference split')
    parser.add_argument(
        '--infer_do_sample', type=int, default=0, help="inference sampling num. greedy if 0")
    parser.add_argument(
        '--infer_num', type=int, default=None, help="limit inference dataset size")
    parser.add_argument(
        '--infer_mode', type=str, default='base', help="inference algorithm")
    parser.add_argument(
        '--infer_no_repeat_size', type=int, default=0, help="no repeat ngram size for inference")
    parser.add_argument(
        '--infer_clip_scale', type=float, default=0.5, help="hyperparam for clip inference")
    parser.add_argument(
        '--infer_ce_scale', type=float, default=0.5, help="hyperparam for clip inference")
    parser.add_argument(
        '--visdial_normalized_loss', action='store_true', default=False, help='normalized loss for visdial')
    parser.add_argument(
        '--visdial_binary_loss', action='store_true', default=False, help='binary loss for visdial')
    parser.add_argument(
        '--visdial_use_yes', action='store_true', default=False, help='use "yes" as the classification label')
    parser.add_argument(
        '--visdial_likelihood_loss', action='store_true', default=False, help='likelihood sum loss')
    parser.add_argument(
        '--infer_force_imagefolder', action='store_true', default=False, help='force image folder')
    parser.add_argument(
        '--infer_prompts', type=str, default=None, help='infer prompts. use fixed_prompt if not set')
    parser.add_argument(
        '--infer_sample_repeat', type=int, default=1, help='infer num samples per image')
    parser.add_argument(
        '--finetune_do_not_normalize_embedding', action='store_true', default=False, help='do not normalize embedding')
    parser.add_argument(
        '--finetune_project_lr', type=float, default=None, help='clip_project learning rate')
    parser.add_argument(
        '--finetune_sample_text', action='store_true', default=False, help='sample text')
    parser.add_argument(
        '--constant_lr', action='store_true', default=False, help='do not decay lr')
    parser.add_argument(
        '--infer_multi_checkpoint', nargs="+", default=[], help='multiple checkpoints to merge')
    parser.add_argument(
        '--load_finetuned', action='store_true', default=False, help='')
    parser.add_argument(
        '--load_mm_policy', action='store_true', default=False, help='')

    # demo
    parser.add_argument(
        '--infer_joint_model', type=str, default='gpt2-medium', help="inference joint lm")
    parser.add_argument(
        '--demo_port', type=int, default=8504, help="port for the demo server")
    parser.add_argument(
        '--demo_joint_model_weight', type=str, default=None, help="demo style generator weight")

    args = parser.parse_args()
    if Path(args.config).is_file():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                print(f'ignoring invalid key from the config file: {k}')
    else:
        args.config = 'none'
    args.cuda = torch.cuda.is_available()

    args.use_pair_reward = not args.do_not_use_pair_reward
    args.use_style_reward = False
    if Path(args.style_cache_path).is_dir():
        if Path(args.label_path).is_file():
            args.use_style_reward = True

    assert (args.use_style_reward | args.use_pair_reward), "Use at least one type of reward"
    if args.use_pair_reward:
        log.info(f'using reward: pair')
    if args.use_style_reward:
        log.info(f'using reward: style')

    if args.init_value_model is None:
        args.init_value_model = args.init_model
    if args.init_value_model_weight == 'INIT':
        args.init_value_model_weight = args.init_model_weight
    elif args.init_value_model_weight == 'None':
        args.init_value_model_weight = None

    if args.style_acc_as_rewards:
        args.style_gain = STYLE_ACC_GAIN
        args.style_bias = STYLE_ACC_BIAS

    if args.use_coco_eval:
        log.info(f'using coco eval')
    if args.use_label_prefix:
        log.info(f'using label prefix')
    num_gpus = torch.cuda.device_count()
    if args.num_gpus is None:
        args.num_gpus = num_gpus
    else:
        args.num_gpus = min(num_gpus, args.num_gpus)

    if args.num_gpus > 1:
        args.save_interval = max(1, args.save_interval // args.num_gpus)
        args.eval_interval = max(1, args.eval_interval // args.num_gpus)
        args.log_interval = max(1, args.log_interval // args.num_gpus)

    if args.checkpoint is not None:
        args.checkpoint = str(Path(args.checkpoint).resolve())
    return args


if __name__ == '__main__':
    args = get_args()
    with open('args_replicate.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
