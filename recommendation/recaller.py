from collections import defaultdict
import faiss
from .model_wraper import NRMSModel, YoutubeNetModel
import json
import os
import numpy as np
from itertools import chain
from gensim.models.keyedvectors import KeyedVectors


class ItemIndex:
    def __init__(self, dimension, encode_func, metric="dot", n_clusters=20, n_pq=10):
        self.dimension = dimension
        self.encode_func = encode_func
        self.metric = metric
        self.item_ids = None
        self.n_clusters = n_clusters
        self.n_pq = n_pq

    def build_index(self, items, save_dir):
        metric = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
        index = faiss.index_factory(self.dimension, f"IVF{self.n_clusters}, PQ{self.n_pq}", metric)
        self.item_ids = [item["id"] for item in items]
        encoded_vecs = self.encode_func(items)
        if self.metric == "cosine":
            faiss.normalize_L2(encoded_vecs)
        index.train(encoded_vecs)
        index.add(encoded_vecs)
        index_file = os.path.join(save_dir, "index.faiss")
        faiss.write_index(index, index_file)
        with open(os.path.join(save_dir, "item_ids.json"), "w") as fo:
            json.dump(self.item_ids, fo)
        self.index = index

    def load_index(self, save_dir):
        index_file = os.path.join(save_dir, "index.faiss")
        self.index = faiss.read_index(index_file)
        with open(os.path.join(save_dir, "item_ids.json")) as fi:
            self.item_ids = json.load(fi)

    def add_items(self, items):
        vecs = self.encode_func(items)
        self.index.add(vecs)
    
    def retrieve(self, query_vecs, topk=10, nprob=4):
        if self.metric == "cosine":
            faiss.normalize_L2(query_vecs)
        _, nns = self.index.search(query_vecs, topk, nprob)
        return [[self.item_ids[idx] for idx in item if idx != -1] for item in nns]


class Recaller:
    def recall(self, clicked_items, user=None, topk=20):
        pass

    def build_index(self, items):
        pass

    def add_items(self, items):
        pass

    def load_index(self):
        pass

    def encode_items(self, items):
        pass


class NRMSRecaller(Recaller):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, "nrms_model.pt")
        self.nrms_model = NRMSModel(model_path)
        self.index = ItemIndex(self.nrms_model.emb_dim, self.encode_items)

    def recall(self, clicked_items, user=None, topk=20):
        history = [item["text"] for item in clicked_items]
        user_vec =  self.nrms_model.encode_users([history])
        item_ids = self.index.retrieve(user_vec, topk)[0]
        clicked_ids = [item["id"] for item in clicked_items]
        recalled_ids = [item_id for item_id in item_ids if item_id not in clicked_ids]
        return recalled_ids
    
    def encode_items(self, items):
        batch_size = 128
        vecs = []
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i+batch_size]
            texts = [item["text"] for item in batch_items]
            batch_vecs = self.nrms_model.encode_docs(texts)
            vecs.append(batch_vecs)
        return np.concatenate(vecs)

    def build_index(self, items):
        self.index.build_index(items, self.model_dir)
    
    def add_items(self, items):
        self.index.add_items(items)
    
    def load_index(self):
        self.index.load_index(self.model_dir)


class Item2vectorRecaller(Recaller):
    def __init__(self, model_dir, last_n=5):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.last_n = last_n
        model_path = os.path.join(model_dir, "item_vecs.txt")
        self.model = KeyedVectors.load_word2vec_format(model_path).wv
        self.index = ItemIndex(self.model.wv.vector_size, self.encode_items, "cosine")

    def recall(self, clicked_items, user=None, topk=20):
        clicked_items = [item for item in clicked_items if item["id"] in self.model.vocab]
        clicked_items[-self.last_n:]
        if not clicked_items:
            return []
        query_vecs = self.encode_items(clicked_items)
        n = topk // self.last_n
        item_ids = self.index.retrieve(query_vecs, n)
        clicked_ids = [item["id"] for item in clicked_items]
        recalled_ids = [item_id for item_id in chain.from_iterable(item_ids) if item_id not in clicked_ids]
        return recalled_ids

    def encode_items(self, items):
        items = [item for item in items if item["id"] in self.model.vocab]
        vecs = [self.model[item["id"]] for item in items]
        return np.array(vecs, dtype="float32")
    
    def build_index(self, items):
        items = [{"id": key} for key in self.model.vocab.keys()]
        self.index.build_index(items, self.model_dir)

    def load_index(self):
        self.index.load_index(self.model_dir)


class YoutubeNetRecaller(Item2vectorRecaller):
    def __init__(self, model_dir, max_history=30, device="cpu"):
        super().__init__(model_dir, 1)
        self.encoder_model = YoutubeNetModel(model_dir, max_history, device=device)

    def recall(self, clicked_items, user=None, topk=20):
        clicked_items = [item["id"] for item in clicked_items if item["id"] in self.model.vocab]
        query_vec = self.encoder_model.encode_user(clicked_items)
        recalled_ids = self.index.retrieve(query_vec, topk)
        return recalled_ids


class CompoundRecaller(Recaller):
    def __init__(self, recallers):
        self.recallers = recallers
    
    def build_index(self, items):
        for recaller in self.recallers:
            recaller.build_index(items)
    
    def load_index(self):
        for recaller in self.recallers:
            recaller.load_index()
    
    def add_items(self, items):
        for recaller in self.recallers:
            recaller.add_items(items)
    
    def recall(self, clicked_items, user=None, topk=20):
        counts = defaultdict(int)
        for recaller in self.recallers:
            for item_id in recaller.recall(clicked_items, topk):
                counts[item_id] += 1
        sorted_items = sorted(counts.items(), lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:topk]]
