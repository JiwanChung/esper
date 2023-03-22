import torch.nn.functional as F


def calc_kl_loss(logits, base_logits, masks):
    masks = masks.float().unsqueeze(-1)
    probs = F.softmax(logits, -1)
    logprobs = F.log_softmax(logits, -1)
    ref_logprobs = F.log_softmax(base_logits, -1)
    kl = probs * (logprobs - ref_logprobs)
    kl = (kl * masks).sum() / masks.sum()
    return kl
