from typing import Counter
import torch
import os.path
import shutil
import lmdb
import pickle
import struct
from tqdm import tqdm
from itertools import chain
import random
import numpy as np


class NRMSDataset(torch.utils.data.Dataset):
    """[summary]

    Args:
        
    """
    def __init__(self, content_file, behavior_file, tokenize, word_dictionary=None, max_history_size=50,
            max_seq_len=25, cache_path=".nrms", rebuild_cache=False, min_tf=3, num_negatives=4):
        super().__init__()
        self.content_file = content_file
        self.behavior_file = behavior_file
        self.tokenize = tokenize
        self.contents = None
        self.contents_converted = False
        self.word_dictionary = word_dictionary
        self.max_history_size = max_history_size
        self.max_seq_len = max_seq_len
        self.min_tf = min_tf
        self.num_negatives = num_negatives
        if rebuild_cache or not os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)
            self._build_cache(cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"] - 1  # account for word_dictionary
            if self.word_dictionary is None:
                self.word_dictionary = pickle.loads(txn.get(b"word_dictionary"))
            self.vocab_size = len(self.word_dictionary) + 2 # account for padding and oov

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            label, history, history_size, history_seq_lens, candidates, candidate_seq_lens = pickle.loads(txn.get(struct.pack(">I", index)))
        return label, history, history_size, history_seq_lens, candidates, candidate_seq_lens

    def __len__(self):
        return self.length

    def _build_cache(self, cache_path):
        if self.word_dictionary is None:
            self._build_word_dictionary()
        if not self.contents_converted:
            self._convert_contents()
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            with env.begin(write=True) as txn:
                txn.put(b"word_dictionary", pickle.dumps(self.word_dictionary))
                for buffer in self._yield_buffer():
                    for key, value in buffer:
                        txn.put(key, value)

    def _load_contents(self):
        with open(self.content_file, encoding="utf-8") as fi:
            headers = fi.readline().strip().split("\t")
            id_idx = headers.index("content_id")
            title_idx = headers.index("text")
            contents = {}
            tbar = tqdm(fi, mininterval=1, smoothing=0.1)
            tbar.set_description("loading contents")
            for line in tbar:
                splits = line.strip().split("\t")
                content_id = splits[id_idx]
                title = splits[title_idx]
                tokens = self.tokenize(title)
                contents[content_id] = tokens
            self.contents = contents
            self.content_converted = False

    def _convert_contents(self):
        if not self.contents:
            self._load_contents()
        oov = len(self.word_dictionary) + 1
        tbar = tqdm(self.contents.keys(), mininterval=1, smoothing=0.1)
        tbar.set_description("converting contents")
        for key in tbar:
            content = self.contents[key]
            content_len = min(self.max_seq_len, len(content))
            content = [self.word_dictionary[token] if token in self.word_dictionary else oov for token in content]
            content = content[:self.max_seq_len] + [0] * max(0, self.max_seq_len - len(content))
            self.contents[key] = (content, content_len)
        self.content_converted = True

    def _build_word_dictionary(self):
        if not self.contents or self.content_converted:
            self._load_contents()
        counts = Counter(chain.from_iterable(self.contents.values()))
        vocabs = [word for word, count in counts.items() if count >= self.min_tf]
        self.word_dictionary = {word: idx+1 for idx, word in enumerate(vocabs)}
    
    def _yield_buffer(self, buffer_size=100):
        buffer = list()
        item_idx = 0
        with open(self.behavior_file) as fi:
            pbar = tqdm(fi, mininterval=1, smoothing=0.1)
            pbar.set_description("creating cache")
            for line in pbar:
                values = line.rstrip("\n").split("\t")
                if len(values) != 5:
                    continue
                history = values[3]
                if history == "":
                    continue
                history = history.split(" ")[-self.max_history_size:]
                history_size = len(history)
                random.shuffle(history)
                click_history = [self.contents[content_id][0] for content_id in history]
                history_seq_lens = [self.contents[content_id][1] for content_id in history]
                click_history += [[0] * self.max_seq_len] * (self.max_history_size - len(click_history))
                history_seq_lens += [1] * (self.max_history_size - len(history_seq_lens))
                history_arrays = np.array(click_history)
                history_seq_lens = np.array(history_seq_lens)

                candidates = values[4].split(" ")
                positives, negatives = self._split_candidates(candidates)
                for labels, candidate_arrays, candidate_seq_lens in self._combine_candidates(positives, negatives):
                    buffer.append((struct.pack(">I", item_idx), pickle.dumps((labels, history_arrays, history_size, history_seq_lens, candidate_arrays, candidate_seq_lens))))
                    item_idx += 1
                    if item_idx % buffer_size == 0:
                        yield buffer
                        buffer.clear()
            yield buffer

    def _split_candidates(self, candidates):
        positives = []
        negatives = []
        for candidate in candidates:
            content_id, label = candidate.split("-")
            if label == "1":
                positives.append(content_id)
            else:
                negatives.append(content_id)
        return positives, negatives

    def _combine_candidates(self, positives, negatives):
        for positive in positives:
            size = min(len(negatives), self.num_negatives)
            sampled_negatives = np.random.choice(negatives, size=size, replace=False)
            samples = [positive] + sampled_negatives.tolist()
            candidates = [self.contents[content_id][0] for content_id in samples]
            candidate_seq_lens = [self.contents[content_id][1] for content_id in samples]
            labels = [1] * len(positives) + [0] * len(sampled_negatives)
            idxes = list(range(len(candidates)))
            random.shuffle(idxes)
            candidates = [candidates[idx] for idx in idxes]
            candidate_seq_lens = [candidate_seq_lens[idx] for idx in idxes]
            labels = [labels[idx] for idx in idxes]
            num_candidates = self.num_negatives + 1
            candidates = candidates[:num_candidates] + [[0] * self.max_seq_len] * max(0, num_candidates - len(candidates))
            candidate_seq_lens = candidate_seq_lens[:num_candidates] + [1] * max(0, num_candidates - len(candidate_seq_lens))
            labels = labels[:num_candidates] + [0] * max(0, num_candidates - len(labels))
            candidate_arrays = np.array(candidates)
            candidate_seq_lens = np.array(candidate_seq_lens)
            labels = np.array(labels)
            yield labels, candidate_arrays, candidate_seq_lens
 