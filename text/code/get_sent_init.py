import json
from itertools import chain
from collections import Counter

# import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

data_path = '../data/texts/corpus.txt'
out_path = '../data/texts/sent_init.json'


'''
class SentSplit:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def split_sentence(self, text):
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        if len(sentences) < 2:
            return [text]
        else:
            sentences = [sentences[0], *[v for v in sentences[1:] if len(v) > 5]]
            return sentences

    def get_sentences(self, texts):
        texts = [[(i, v2) for v2 in self.split_sentence(v)] for i, v in enumerate(texts)]
        texts = list(chain(*texts))
        ids, texts = zip(*texts)
        return ids, texts
'''


# sent_split = SentSplit()
tok = AutoTokenizer.from_pretrained('gpt2')
data = []
with open(data_path) as f:
    for line in tqdm(f):
        data.append(line.strip())

c = Counter()
for row in tqdm(data, total=len(data)):
    # _, sent_texts = sent_split.get_sentences(row)
    sent_texts = row.split('. ')
    sents = [v.strip() for v in sent_texts if len(v) > 10]
    tokens = [v.split()[0] for v in sents]
    ids = [tok.encode(v[:10])[0] for v in sents]
    tokens = tok.convert_ids_to_tokens(ids)
    c.update(tokens)

print(f'most_freq: {c.most_common(10)}')
total = sum(c.values())
freq = {k: v / total for k, v in c.items()}
with open(out_path, 'w') as f:
    json.dump(freq, f, indent=4)
