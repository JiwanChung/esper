from transformers import AutoTokenizer


def get_tokenizer(gpt_name):
    # use_fast = False to use slow ver for compatibility with the torch code
    tokenizer = AutoTokenizer.from_pretrained(gpt_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pre_vocab_size = len(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if 'gpt' in gpt_name:
        tokenizer.whitespace_char = chr(288)
    else:
        tokenizer.whitespace_char = chr(9601)
    tokenizer = add_control_tokens(tokenizer, gpt_name)
    return tokenizer


def add_control_tokens(tokenizer, lm_model_name):
    new_tokens = ['[SOS]', '[GEN]', '[EOS]']
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    return tokenizer
