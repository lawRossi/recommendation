from collections import defaultdict
from torch.utils import data
import lmdb
import shutil
import pickle
import struct
import os
import numpy as np
from tqdm import tqdm
import random
import torch


class RecallDataset(data.Dataset):
    def __init__(self,click_list_file, attr_file, min_count):
        self.click_list_file = click_list_file
        self.attr_file = attr_file
        self.min_count = min_count
    
    def _build_vocabulary(self):
        with open(self.click_list_file) as fi:
            counts = defaultdict(int)
            for line in fi:
                _, clicked_ids = line.strip().split("\t")
                for clicked_id in clicked_ids.split(" "):
                    counts[clicked_id] += 1
            counts = {k: v for k, v in counts.items() if v >= self.min_count}
            ids, freqs = zip(*counts.items())
            content_ids = np.array(ids)
            freqs = [0] + list(freqs)  # add frequence for padding idx
            freqs = np.array(freqs) ** 0.75
            freqs = freqs / freqs.sum()
            vocab = {content_id: i + 1 for i, content_id in enumerate(content_ids)}
            self.vocab = vocab
            self.weights = freqs
        if self.attr_file:
            attr_counts = defaultdict(lambda : defaultdict(int))
            for attrs in self.attributes.values():
                for attr_name in self.attribute_names:
                    attr_counts[attr_name][attrs[attr_name]] += 1
            self.attr_vocabs = {}
            for attr_name, counts in attr_counts.items():
                attr_vocab = {attr for attr, count in counts.items() if count >= self.min_count}
                self.attr_vocabs[attr_name] = {attr: i for i, attr in enumerate(attr_vocab)}
        else:
            self.attr_vocabs = None

    def _load_attributes(self):
        with open(self.attr_file, encoding="utf-8") as fi:
            headers = fi.readline().strip().split("\t")[1:]  # skip id
            self.attribute_names = headers
            self.attributes = {}
            for line in fi:
                splits = line.strip().split("\t")
                self.attributes[splits[0]] = (dict(zip(headers, splits[1:])))

    def _build_cache(self, cache_path):
        if self.attr_file:
            self._load_attributes()
        self._build_vocabulary()
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            with env.begin(write=True) as txn:
                txn.put(b"components", pickle.dumps((self.vocab, self.attr_vocabs, self.weights)))
                for buffer in self._yield_buffer():
                    for key, value in buffer:
                        txn.put(key, value)


class YoutubeNetDataset(RecallDataset):
    def __init__(self, click_list_file, attr_file=None, min_history=3, max_history=30, 
            cache_path=".recall", rebuild_cache=False, min_count=3):
        super().__init__(click_list_file, attr_file, min_count)
        self.min_history = min_history
        self.max_history = max_history
        if not os.path.exists(cache_path) or rebuild_cache:
            shutil.rmtree(cache_path, ignore_errors=True)
            self._build_cache(cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"] - 1
            vocab, attr_vocabs, weights = pickle.loads(txn.get(b"components"))
            self.vocab = vocab
            self.attr_vocabs = attr_vocabs
            self.weights = weights
        if self.attr_vocabs:
            self._load_attributes()

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index: int):
        with self.env.begin(write=False) as txn:
            user_id, history, positive = pickle.loads(txn.get(struct.pack(">I", index)))
            history = [self.vocab.get(content_id, len(self.vocab)+1) for content_id in history]
            history_size = len(history)
            history = history + [0] * (self.max_history - len(history))
            positive = self.vocab.get(positive)
            values = [np.array(history), history_size, positive]
            if self.attr_vocabs:
                attrs = self.attributes.get(user_id)
                attr_values = [vocab.get(attrs[attr_name], len(vocab)+1) for attr_name, vocab in self.attr_vocabs.items()]
                values.extend(attr_values)
            return tuple(values)

    def _yield_buffer(self, buffer_size=10000):
        buffer = list()
        idx = 0
        with open(self.click_list_file) as fi:
            pbar = tqdm(fi, mininterval=1, smoothing=0.1)
            pbar.set_description("building cache")
            for line in pbar:
                user_id, clicked_ids = line.strip().split("\t")
                clicked_ids = clicked_ids.split(" ")
                for i in range(self.min_history, len(clicked_ids)):
                    positive = clicked_ids[i]
                    if positive not in self.vocab:
                        continue
                    history = clicked_ids[max(0, i-self.max_history):i]
                    buffer.append((struct.pack(">I", idx), pickle.dumps((user_id, history, positive))))
                    idx += 1
                    if len(buffer) == buffer_size:
                        yield buffer
                        buffer.clear()
            yield buffer


class GESDataset(RecallDataset):
    """[summary]
    """
    def __init__(self, click_history_file, attr_file, window_size=5, cache_path=".ges", rebuild_cache=False,
            min_count=3,  num_negative=5):
        super().__init__(click_history_file, attr_file, min_count)
        self.window_size = window_size
        self.num_negative = num_negative

        if rebuild_cache or not os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)
            self._build_cache(cache_path)

        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"] - 1
            vocab, attr_vocabs, weights = pickle.loads(txn.get(b"components"))
            self.vocab = vocab
            self.attr_vocabs = attr_vocabs
            self.items = np.arange(0, len(self.vocab)+1)
            self.weights = torch.tensor(weights)
        if self.attr_vocabs:
            self._load_attributes()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            central_id, context = pickle.loads(txn.get(struct.pack(">I", index)))
            negatives = torch.multinomial(self.weights, len(context)*self.num_negative, replacement=True)
            negatives = [self.items[idx] for idx in negatives]
            labels = [1] * len(context) + [0] * len(negatives)
            context = [self.vocab.get(item_id, len(self.vocab)+1) for item_id in context]
            context.extend(negatives)
            idxes = list(range(len(context)))
            random.shuffle(idxes)
            context = np.array([context[idx] for idx in idxes])
            labels = np.array([labels[idx] for idx in idxes])

            central_idx = self.vocab[central_id]
            values = [labels, central_idx, context]

            attrs = []
            for attr_name in self.attribute_names:
                attr_value = self.attributes[central_id].get(attr_name)
                attr_vocab = self.attr_vocabs[attr_name]
                attrs.append(attr_vocab.get(attr_value, len(attr_vocab)))
            values.extend(attrs)

            return tuple(values)

    def _yield_buffer(self, buffer_size=1000):
        buffer = list()
        idx = 0
        with open(self.click_list_file) as fi:
            pbar = tqdm(fi, mininterval=1, smoothing=0.1)
            pbar.set_description("building cache")
            for line in pbar:
                _, clicked_ids = line.strip().split("\t")
                clicked_ids = clicked_ids.split(" ")
                for i in range(self.window_size, len(clicked_ids)-self.window_size):
                    central_id = clicked_ids[i]
                    if central_id not in self.vocab:
                        continue
                    context = clicked_ids[i-self.window_size:i] + clicked_ids[i+1:i+self.window_size]
                    buffer.append((struct.pack(">I", idx), pickle.dumps((central_id, context))))
                    idx += 1
                    if len(buffer) == buffer_size:
                        yield buffer
                        buffer.clear()
            yield buffer
