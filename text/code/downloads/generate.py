from pathlib import Path

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import root

num_samples = 150000
prefix_len = 20


out_dir = root / 'data/texts'
Path(out_dir).mkdir(exist_ok=True)

device = 'cuda'
transformer = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(transformer).to(device)
tokenizer = AutoTokenizer.from_pretrained(transformer)
tokenizer.pad_token = tokenizer.eos_token

i = 1
prefixes = []
with open(Path(out_dir) / 'corpus.txt', 'r') as f:
    for line in f:
        prefixes.append(line.strip()[:prefix_len])
        if i >= num_samples:
            break
        i += 1


res = []
for prefix in tqdm(prefixes, total=len(prefixes)):
    tokens = tokenizer(prefix, return_tensors='pt').to(device)
    text = model.generate(**tokens, do_sample=True, top_p=0.9, max_length=80,
                          pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(text[0])
    text = '.'.join(text.split('.')[1:]).strip()
    text = text.replace('<|endoftext|>', '')
    text = text.replace('\n', ' ')
    if text:
        res.append(text)


print(f"{len(res)} lines in total")
with open(Path(out_dir) / 'generation.txt', 'w') as f:
    for line in res:
        f.write(f'{line}\n')
