import re
import random
import json
import pickle
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import torch
from transformers import AutoTokenizer


def get_label_dataset(labels):
    # labels: {name: index}
    return [{'label': k, 'text': ''} for k in labels.keys()]


class Dataset(torch.utils.data.Dataset):
    label_map = {
        'twitter': 'sns',
        'instagram': 'sns',
        'goodnews': 'news',
        'visdial': 'dialogue'
    }
    @classmethod
    def splits(cls, data_dir='../data/texts', splits=[8,1,1], max_num_lines=None,
               filenames=None, is_generator=False, add_eot=False):
        data_dir = Path(data_dir)
        data, length_per_cat, labels = cls.load_data(data_dir, max_num_lines, filenames)
        print(f"length_per_cat: {length_per_cat}")
        print(f"total_length: {len(data)}")
        cut = length_per_cat // sum(splits)
        splitted = {'train': [], 'val': [], 'test': []}
        length_per_cats = {}
        if is_generator:
            splitted['train'] = data
            length_per_cats['train'] = length_per_cat
            label_dataset = get_label_dataset(labels)
            splitted['val'] = label_dataset
            splitted['test'] = label_dataset
            length_per_cats['val'] = 1
            length_per_cats['test'] = 1
        else:
            for i in range(len(data) // length_per_cat):
                subset = data[i * length_per_cat: (i + 1) * length_per_cat]
                train = subset[:-sum(splits[1:]) * cut]
                val = subset[-sum(splits[1:]) * cut: -splits[-1] * cut]
                test = subset[-splits[-1] * cut:]
                splitted['train'].extend(train)
                splitted['val'].extend(val)
                splitted['test'].extend(test)
                length_per_cats['train'] = len(train)
                length_per_cats['val'] = len(val)
                length_per_cats['test'] = len(test)
        name = cls.get_name(filenames)
        print(f"cache_name: {name}")
        with open(data_dir / f'labels_{name}.json', 'w') as f:
            json.dump(labels, f, indent=4)

        return {k: cls(data_dir, v, length_per_cats[k], labels, split=k, add_eot=add_eot) for k, v in splitted.items()}

    def __init__(self, data_dir='../data/texts',
                 data=None, length_per_cat=None, labels=None,
                 max_num_lines=10000, split='train', add_eot=False):
        self.add_eot = add_eot
        self.eot = '<|endoftext|>'
        self.split = split
        self.data_dir = Path(data_dir)
        if data is not None:
            self.data = data
            self.length_per_cat = length_per_cat
            self.labels = labels
        else:
            self.data, self.length_per_cat, self.labels = self.load_data(self.data_dir, max_num_lines)
            print(f"length_per_cat: {self.length_per_cat}")
            print(f"total_length: {len(self.data)}")

    @classmethod
    def get_name(self, filenames):
        name = 'audio_all' if filenames is None else '_'.join(filenames)
        return name

    @classmethod
    def load_data(cls, data_dir, max_num_lines=None, filenames=None):
        name = cls.get_name(filenames)
        cache_path = data_dir / 'cache' / f'{name}.pkl'
        cache_path.parent.mkdir(exist_ok=True)
        if not cache_path.is_file():
            data, length_per_cat, labels = cls._load_data(data_dir, max_num_lines, filenames)
            x = {'data': data, 'length_per_cat': length_per_cat, 'labels': labels}
            with open(cache_path, 'wb') as f:
                pickle.dump(x, f)
        else:
            with open(cache_path, 'rb') as f:
                x = pickle.load(f)
            data, length_per_cat, labels = x['data'], x['length_per_cat'], x['labels']
        return data, length_per_cat, labels

    @classmethod
    def _load_data(cls, data_dir, max_num_lines=None, filenames=None):
        random.seed(0)
        if filenames is None:
            paths = sorted(list(data_dir.glob('*.txt')))
        else:
            paths = sorted([data_dir / f'{n}.txt' for n in filenames])
        data = defaultdict(lambda: [])
        pbar = tqdm(paths, total=len(paths), desc="loading text files")
        for path in pbar:
            _name = path.stem.split('_')[0]  # e.g. twitter
            name = cls.label_map.get(_name, _name)
            with open(path) as f:
                for i, line in enumerate(f):
                    if max_num_lines is not None:
                        if i == max_num_lines:
                            break
                    line = cls.preprocess(line.strip())
                    if len(line) > 15:
                        data[name].append(line)
        length_per_cat = min([len(v) for v in data.values()])
        print({k: len(v) for k, v in data.items()})
        flattened = []
        for k, v in tqdm(data.items(), total=len(data), desc="flattening"):
            flat = random.sample(v, length_per_cat)
            for line in flat:
                flattened.append({'label': k, 'text': line})
        labels = {v: i for i, v in enumerate(sorted(list(data.keys())))}
        return flattened, length_per_cat, labels

    def __len__(self):
        return len(self.data)

    @staticmethod
    def preprocess(string):
        string = string.replace("<|LB|>", "\n")  # line break
        # mostly from
        # https://github.com/kingoflolz/mesh-transformer-jax/blob/master/create_finetune_tfrecords.py#L98
        string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
        string = string.replace(" @-@ ", "-")
        string = string.replace(" @,@ ", ",")
        string = string.replace(" @.@ ", ".")
        string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
        string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
        string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
        string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
        string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
        string = string.replace(" \n", "\n")
        string = string.replace("\n ", "\n")
        return string

    def __getitem__(self, index):
        row = self.data[index]
        if self.add_eot:
            row['text'] = row['text'].strip() + self.eot
        row['label_index'] = self.labels[row['label']]
        return row


class Collator:
    def __init__(self, transformer, max_length=128, is_generator=False, prefix_label=True):
        self.max_length = max_length
        self.prefix_label = prefix_label
        self.is_generator = is_generator
        self.tokenizer = AutoTokenizer.from_pretrained(transformer, pad_token='<|endoftext|>')

    def __call__(self, batch):
        inputs = [row['text'] for row in batch]
        text_labels = [row['label'] for row in batch]
        if self.is_generator:
            if self.prefix_label:
                inputs = [f'{label.lower()}: {text.strip()}' if text else f'{label.lower()}:'
                        for label, text in zip(text_labels, inputs)]
            else:
                inputs = [text.strip() for text in inputs]
        inputs = self.tokenizer(inputs, return_tensors='pt', truncation=True,
                                padding=True,
                                max_length=self.max_length,
                                add_special_tokens=False)
        labels = torch.Tensor([row['label_index'] for row in batch]).long()
        batch = {**inputs, 'labels': labels, 'text_labels': text_labels}
        return batch


def get_loaders(data_dir='../data/texts', transformer='bert-base-uncased', max_length=128,
                add_eot=False,
                is_generator=False, prefix_label=True, filenames=None, batch_size=32, num_workers=32):
    data = Dataset.splits(filenames=filenames, is_generator=is_generator, add_eot=add_eot)
    collate = Collator(transformer, max_length, is_generator, prefix_label)
    loaders = {k: torch.utils.data.DataLoader(v, collate_fn=collate,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              shuffle=(k == 'train'))
               for k, v in data.items()}
    return loaders


if __name__ == '__main__':
    loaders = get_loaders()
    for batch in loaders['train']:
        x = batch
        break
