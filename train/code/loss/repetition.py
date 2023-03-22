import torch
import numpy as np

from replay import count_rep


def get_whitespace_repetition_penalty(texts):
    rep_scores = np.array([count_rep(v, 2) for v in texts])
    rep_scores = 1 / (rep_scores + 1)
    rep_scores_1 = np.array([count_rep(v, 1) for v in texts])
    rep_scores_1 = 1 / (rep_scores_1.clip(min=2))
    rep_scores = rep_scores ** 2 + rep_scores_1
    return rep_scores


def get_gpt_repetition_penalty(texts, tokenizer, weight=2):
    scores = []
    for text in texts:
        tokens = torch.tensor(tokenizer.encode(text)).long()
        z = torch.stack([tokens, tokens.roll(-1, 0), tokens.roll(-2, 0)], dim=1)
        score3 = (len(z) - len(z.unique(dim=0))) * (weight ** 2)
        score2 = (len(z) - len(z[:, :2].unique(dim=0))) * (weight)
        score1 = min((len(z) - len(z[:, :1].unique(dim=0)) - 1), 0)  # one 1-gram repetition is ok
        score = score1 + score2 + score3
        scores.append(score)
    scores = np.array(scores)
    return -scores
