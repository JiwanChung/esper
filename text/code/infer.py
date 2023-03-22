import json

import transformers
from transformers import AutoTokenizer

from train import Model


transformer = 'bert-base-cased'
max_length = 128
device = 'cuda'
ckpt_path = '../data/log/25-Mar-2022--15-06-38/default/version_0/checkpoints/best.ckpt'
label_path = '../data/texts/labels.json'

transformers.trainer_utils.set_seed(0)

with open(label_path) as f:
    labels = json.load(f)
labels = {i: v for v, i in labels.items()}

model = Model.load_from_checkpoint(ckpt_path)
model.eval()
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(transformer)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def run_sample(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True,
                     max_length=max_length)
    outputs = model.model(**inputs.to(device))
    logits = outputs.logits
    index = logits.argmax(-1).item()
    label = labels[index]
    return label


x = run_sample('Thanks @a16z for naming @huggingface as one of the top data startups in the world ')
print(x)
import ipdb; ipdb.set_trace()  # XXX DEBUG
