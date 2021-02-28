from collections import defaultdict
import faiss
from model import NRMSModel
import json
import os
import numpy as np
from itertools import chain
from gensim.models.keyedvectors import KeyedVectors


class ContentIndex:
    def __init__(self, dimension, encode_func, metric="dot"):
        self.dimension = dimension
        self.encode_func = encode_func
        self.metric = metric
        self.content_ids = None
    
    def build_index(self, contents, save_dir):
        index = faiss.index_factory(self.dimension, "IVF100, PQ10", faiss.METRIC_L2)
        self.content_ids = [content["id"] for content in contents]
        encoded_vecs = self.encode_func(contents)
        if self.metric == "cosine":
            faiss.normalize_L2(encoded_vecs)
        index.train(encoded_vecs)
        index.add(encoded_vecs)
        index_file = os.path.join(save_dir, "index.faiss")
        faiss.write_index(index, index_file)
        with open(os.path.join(save_dir, "content_ids.json"), "w") as fo:
            json.dump(self.content_ids, fo)
    
    def load_index(self, save_dir):
        index_file = os.path.join(save_dir, "index.faiss")
        self.index = faiss.read_index(index_file)
        with open(os.path.join(save_dir, "content_ids.json")) as fi:
            self.content_ids = json.load(fi)

    def add_contents(self, contents):
        vecs = self.encode_func(contents)
        self.index.add(vecs)
    
    def retrieve(self, query_vecs, topk=10):
        if self.metric == "cosine":
            faiss.normalize_L2(query_vecs)
        _, nns = self.index.search(query_vecs, topk)
        return [[self.content_ids[idx] for idx in item if idx != -1] for item in nns]


class Recaller:
    def recall(self, clicked_contents, user=None, topk=20):
        pass

    def build_index(self, contents):
        pass

    def add_contents(self, contents):
        pass

    def load_index(self):
        pass

    def encode_contents(self, contents):
        pass


class NRMSRecaller(Recaller):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        vocab_file = os.path.join(model_dir, "vocab.json")
        model_path = os.path.join(model_dir, "nrms_model.pt")
        self.nrms_model = NRMSModel(model_path, "emb_dims")
        self.index = ContentIndex(self.nrms_model.emb_dim, self.encode_contents)
    
    def recall(self, clicked_contents, user=None, topk=20):
        history = [content["text"] for content in clicked_contents]
        user_vec =  self.nrms_model.encode_users([history])
        content_ids = self.index.retrieve(user_vec, topk)[0]
        clicked_ids = [content["id"] for content in clicked_contents]
        recalled_ids = [content_id for content_id in content_ids if content_id not in clicked_ids]
        return recalled_ids
    
    def encode_contents(self, contents):
        batch_size = 128
        vecs = []
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i+batch_size]
            texts = [content["text"] for content in batch_contents]
            batch_vecs = self.nrms_model.encode_docs(texts)
            vecs.append(batch_vecs)
        return np.concatenate(vecs)

    def build_index(self, contents):
        self.index.build_index(contents, self.model_dir)
    
    def add_contents(self, contents):
        self.index.add_contents(contents)
    
    def load_index(self):
        self.index.load_index(self.model_dir)


class Item2vectorRecaller(Recaller):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, "item_vecs.txt")
        self.model = KeyedVectors.load_word2vec_format(model_path).wv
        self.index = ContentIndex(self.model.wv.vector_size, self.encode_contents, "cosine")

    def recall(self, clicked_contents, user=None, topk=20):
        clicked_contents = [content for content in clicked_contents if content["id"] in self.model.vocab]
        if not clicked_contents:
            return []
        query_vecs = self.encode_contents(clicked_contents)
        content_ids = self.index.retrieve(query_vecs, topk)
        clicked_ids = [content["id"] for content in clicked_contents]
        recalled_ids = [content_id for content_id in chain.from_iterable(content_ids) if content_id not in clicked_ids]
        return recalled_ids

    def encode_contents(self, contents):
        contents = [content for content in contents if content["id"] in self.model.vocab]
        vecs = [self.model[content["id"]] for content in contents]
        return np.array(vecs, dtype="float32")
    
    def build_index(self, contents):
        contents = [{"id": key} for key in self.model.vocab.keys()]
        self.index.build_index(contents, self.model_dir)
    
    def load_index(self):
        self.index.load_index(self.model_dir)


class CompoundRecaller(Recaller):
    def __init__(self, recallers):
        self.recallers = recallers
    
    def build_index(self, contents):
        for recaller in self.recallers:
            recaller.build_index(contents)
    
    def load_index(self):
        for recaller in self.recallers:
            recaller.load_index()
    
    def add_contents(self, contents):
        for recaller in self.recallers:
            recaller.add_contents(contents)
    
    def recall(self, clicked_contents, user=None, topk=20):
        counts = defaultdict(int)
        for recaller in self.recallers:
            for content_id in recaller.recall(clicked_contents, topk):
                counts[content_id] += 1
        sorted_items = sorted(counts.items(), lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:topk]]
