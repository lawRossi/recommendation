from collections import defaultdict
from numpy.core.arrayprint import set_string_function
from torch.utils import data
import lmdb
import shutil
import pickle
import struct
import os
import numpy as np
from tqdm import tqdm


class RecallDataset(data.Dataset):
    def __init__(self, click_list_file, attr_file=None, min_history=3, max_history=30, 
            cache_path=".recall", rebuild_cache=False, min_count=3):
        super().__init__()
        self.click_list_file = click_list_file
        self.attr_file = attr_file
        self.min_history = min_history
        self.max_history = max_history
        self.min_count = min_count
        if not os.path.exists(cache_path) or rebuild_cache:
            shutil.rmtree(cache_path, ignore_errors=True)
            self._build_cache(cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"] - 1
            content_ids, vocab, attr_vocabs, freqs = pickle.loads(txn.get(b"components"))
            self.content_ids = content_ids
            self.vocab = vocab
            self.attr_vocabs = attr_vocabs
            self.content_freqs = freqs

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

    def _load_attributes(self):
        with open(self.attr_file, encoding="utf-8") as fi:
            headers = fi.readline().split("\t")[1:]  # skip id
            self.attribute_names = headers
            self.attributes = {}
            for line in fi:
                splits = line.strip().split("\t")
                self.attributes[splits[0]] = (dict(zip(headers, splits[1:])))

    def _build_vocab_builary(self):
        with open(self.click_list_file) as fi:
            counts = defaultdict(int)
            for line in fi:
                _, clicked_ids = line.strip().split("\t")
                for clicked_id in clicked_ids.split(" "):
                    counts[clicked_id] += 1
            counts = {k: v for k, v in counts.items() if v >= self.min_count}
            ids, freqs = zip(*counts.items())
            content_ids = np.array(ids)
            freqs = np.array(freqs) ** 0.75
            freqs = freqs / freqs.sum()
            vocab = {content_id: i + 1 for i, content_id in enumerate(content_ids)}
            self.content_ids = content_ids
            self.vocab = vocab
            self.content_freqs = freqs
        if self.attr_file:
            attr_counts = defaultdict(lambda : defaultdict(int))
            for attrs in self.attributes.values():
                for attr_name in self.attribute_names:
                    attr_counts[attr_name][attrs[attr_name]] += 1
            self.attr_vocabs = {
                attr_name: {k: i for i, (k, v) in enumerate(counts.items()) if v >= self.min_count} 
                for attr_name, counts in attr_counts.items()
            }
        else:
            self.attr_vocabs = None

    def _build_cache(self, cache_path):
        if self.attr_file:
            self._load_attributes()
        self._build_vocab_builary()
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            with env.begin(write=True) as txn:
                txn.put(b"components", pickle.dumps((self.content_ids, self.vocab, self.attr_vocabs, self.content_freqs)))
                for buffer in self._yield_buffer():
                    for key, value in buffer:
                        txn.put(key, value)

    def _yield_buffer(self, buffer_size=10000):
        buffer = list()
        idx = 0
        with open(self.click_list_file) as fi:
            pbar = tqdm(fi, mininterval=1, smoothing=0.1)
            pbar.set_description("building cache")
            for line in fi:
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
                if idx >= 1000:
                        break
            yield buffer

