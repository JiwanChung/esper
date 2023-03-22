import numpy as np

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


def get_length(dt):
    return np.array([len(v[0].split()) for v in dt.values()]).mean()


def do_cuts(dt, eos_token):
    return {k: [do_cut(v2, eos_token) for v2 in v] for k, v in dt.items()}

def do_cut(text, eos_token):
    if eos_token in text:
        idx = text.find(eos_token)
        # include eos_token
        text = text[:idx + 1]
    return text


class Eval:
    def __init__(self, use_spice=False):
        self.tokenizer = PTBTokenizer()
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
        if use_spice:
            self.scorers.append(
                (Spice(), "SPICE")
            )

    def format_captions(self, x):
        if isinstance(x, str):
            return [{'caption': x}]
        else:
            return [{'caption': v} for v in x]

    def __call__(self, res, gts, cut=False):
        stats = self.run(res, gts, cut)
        return stats

    def run(self, res, gts, cut=False):
        gts = {i: self.format_captions(v) for i, v in enumerate(gts)}
        res = {i: self.format_captions(v) for i, v in enumerate(res)}

        gts = self.tokenizer.tokenize(gts)
        res = self.tokenizer.tokenize(res)

        stats = {}

        stats['gts_len'] = get_length(gts)
        if cut:
            length = int(stats['gts_len'])
            res = {k: [' '.join(v[0].split()[:length])] for k, v in res.items()}
        stats['hypo_len'] = get_length(res)

        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    stats[m] = sc
            else:
                stats[method] = score

        return stats
