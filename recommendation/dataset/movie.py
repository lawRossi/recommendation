from torch.utils import data
import shutil
import os
import lmdb
import pickle
import struct
from tqdm import tqdm
import json
import numpy as np
from itertools import chain
from collections import Counter
import random
import torch


class MovieDataset(data.Dataset):
    def __init__(self, movie_file, rating_file, max_celebrities=10, max_tags=5, min_count=3, cache_path=".movie"):
        self.movie_file = movie_file
        self.rating_file = rating_file
        self.max_celebrities = max_celebrities
        self.max_tags = max_tags
        self.cache_path = cache_path
        self.min_count = min_count

    def _convert_movie(self, movie_id):
        movie_idx = self.movie_vocab[movie_id]
        celebrities = [self.celebrity_vocab.get(celebrity, self.celebrity_oov) for celebrity in self.movies[movie_id]["celebrities"]]
        celebrities = celebrities[:self.max_celebrities] + [0] * max(0, self.max_celebrities - len(celebrities))
        celebrities = np.array(celebrities)
        tags = [self.tag_vocab.get(tag, self.tag_oov) for tag in self.movies[movie_id]["tags"]]
        tags = tags[:self.max_tags] + [0] * max(0, self.max_tags - len(tags))
        tags = np.array(tags)
        return movie_idx, celebrities, tags

    def __len__(self):
        return self.length

    def _load_ratings(self):
        print("loading ratings")
        with open(self.rating_file, encoding="utf-8") as fi:
            self.ratings = {}
            rating_data = json.load(fi)
            for user_id, ratings in rating_data.items():
                ratings = sorted(ratings, key=lambda rating: rating["date"])
                likes = [str(rating["movie_id"]) for rating in ratings if rating["rating"] > 3]
                dislikes = [str(rating["movie_id"]) for rating in ratings if rating["rating"] <= 3]
                self.ratings[user_id] = {"likes": likes, "dislikes": dislikes}

    def _load_movies(self):
        print("loading movies")
        rated_movies = set()
        for ratings in self.ratings.values():
            rated_movies.update(ratings["likes"] + ratings["dislikes"])

        with open(self.movie_file, encoding="utf-8") as fi:
            movies = [json.loads(line) for line in fi]
            self.movies = {}
            for movie in movies:
                movie_id = movie["id"]
                if movie_id in rated_movies:
                    celebrities = [item["celebrity_id"] for item in movie["directors"]]
                    for item in movie["screenWriters"] + movie["starrings"]:
                        if item["celebrity_id"] not in celebrities:
                            celebrities.append(item["celebrity_id"])
                    celebrities = [celebrity for celebrity in celebrities if celebrity is not None and len(celebrity) >= 5]
                    celebrities = celebrities[:self.max_celebrities]
                    tags = movie.get("producing_countries", []) + movie["genres"]
                    for tag in movie["tags"]:
                        if tag not in tags:
                            tags.append(tag)
                    tags = [tag.replace(" ", "") for tag in tags]
                    self.movies[movie_id] = {"celebrities":celebrities, "tags": tags}

    def build_vocabularies(self):
        counts = Counter(chain.from_iterable(value["likes"] for value in self.ratings.values()))
        items, freqs = zip(*[(movie_id, count) for movie_id, count in counts.items() if count >= self.min_count])
        self.movie_vocab = {movie_id: i for i, movie_id in enumerate(items)}
        freqs = np.array(freqs) ** 0.75
        freqs = freqs / freqs.sum()
        self.freqs = freqs
    
        celebrity_counts = Counter(chain.from_iterable([movie["celebrities"] for movie in self.movies.values()]))
        celebrities = [celebrity for celebrity, count in celebrity_counts.items() if count >= self.min_count]
        self.celebrity_vocab = {celebrity_id: i + 1 for i, celebrity_id in enumerate(celebrities)}

        tag_counts = Counter(chain.from_iterable([movie["tags"] for movie in self.movies.values()]))
        tags = [tag for tag, count in tag_counts.items() if count >= self.min_count]
        self.tag_vocab = {tag: i + 1 for i, tag in enumerate(tags)}
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        with open(os.path.join(self.cache_path, "vocab.pkl"), "wb") as fo:
            pickle.dump((self.movie_vocab, self.celebrity_vocab, self.tag_vocab), fo)
        with open(os.path.join(self.cache_path, "freqs.pkl"), "wb") as fo:
                pickle.dump(self.freqs, fo)

    def _build_cache(self, cache_path):
        self._load_ratings()
        self._load_movies()
        self.build_vocabularies()
        with open(os.path.join(cache_path, "movie.pkl"), "wb") as fo:
            pickle.dump(self.movies, fo)
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            with env.begin(write=True) as txn:
                for buffer in self._yield_buffer():
                    for key, value in buffer:
                        txn.put(key, value)


class SkipGramDataset(MovieDataset):
    def __init__(self, rating_file, window_size=3, cache_path="data/.skipgram", rebuild_cache=False, max_ratings=100, 
            min_count=3, num_negative=5) -> None:
        super().__init__(None, rating_file, None, None, min_count, cache_path)
        self.window_size =window_size
        self.max_ratings = max_ratings
        self.num_negative = num_negative

        if rebuild_cache or not os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)
            self._build_cache(cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
            with open(os.path.join(cache_path, "vocab.pkl"), "rb") as fi:
                self.movie_vocab = pickle.load(fi)
            with open(os.path.join(cache_path, "freqs.pkl"), "rb") as fi:
                self.freqs = pickle.load(fi)
            self.all_movies = np.array(range(len(self.movie_vocab)))

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
         with self.env.begin(write=False) as txn:
            central_id, context = pickle.loads(txn.get(struct.pack(">I", index)))
            central_idx = self.movie_vocab[central_id]
            context = [self.movie_vocab[movie_id] for movie_id in context]
            negatives = np.random.choice(self.all_movies, size=len(context)*self.num_negative, p=self.freqs)
            labels = [1] * len(context) + [0] * len(negatives)
            context.extend(negatives)
            idxes = list(range(len(context)))
            random.shuffle(idxes)
            context = [context[idx] for idx in idxes]
            labels = [labels[idx] for idx in idxes]
            return central_idx, np.array(context), np.array(labels)

    def _load_ratings(self):
        print("loading ratings")
        with open(self.rating_file, encoding="utf-8") as fi:
            self.ratings = {}
            rating_data = json.load(fi)
            for user_id, ratings in rating_data.items():
                ratings = sorted(ratings, key=lambda rating: rating["date"])
                likes = [str(rating["movie_id"]) for rating in ratings if rating["rating"] > 3]
                likes = likes[:self.max_ratings]
                dislikes = [str(rating["movie_id"]) for rating in ratings if rating["rating"] <= 3]
                self.ratings[user_id] = {"likes": likes, "dislikes": dislikes}

    def build_vocabularies(self):
        counts = Counter(chain.from_iterable(value["likes"] for value in self.ratings.values()))
        items, freqs = zip(*[(movie_id, count) for movie_id, count in counts.items() if count >= self.min_count])
        self.movie_vocab = {movie_id: i for i, movie_id in enumerate(items)}
        freqs = np.array(freqs) ** (3 / 4)
        freqs = freqs / freqs.sum()
        self.freqs = freqs
        os.makedirs(self.cache_path)
        with open(os.path.join(self.cache_path, "vocab.pkl"), "wb") as fo:
            pickle.dump(self.movie_vocab, fo)
        with open(os.path.join(self.cache_path, "freqs.pkl"), "wb") as fo:
            pickle.dump(self.freqs, fo)

    def _load_movies(self):
        pass

    def _yield_buffer(self, buffer_size=1000):
        buffer = list()
        item_idx = 0
        pbar = tqdm(self.ratings.values(), mininterval=1, smoothing=0.1)
        pbar.set_description("processing ratings")
        for rating in pbar:
            likes = rating["likes"]
            likes = [movie_id for movie_id in likes if movie_id in self.movie_vocab]
            for i in range(self.window_size, len(likes)-self.window_size):
                central_id = likes[i]
                context = likes[i-self.window_size:i] + likes[i+1:i+self.window_size]
                buffer.append((struct.pack(">I", item_idx), pickle.dumps((central_id, context))))
                item_idx += 1
                if len(buffer) == buffer_size:
                    yield buffer
                    buffer.clear()
        yield buffer


class GESDataset(MovieDataset):
    """[summary]
    """
    def __init__(self, movie_file, rating_file, window_size=3, cache_path="data/.movie_embs", rebuild_cache=False, max_ratings=100, 
            min_count=3,  num_negative=5):
        super().__init__(movie_file, rating_file, None, None, min_count, cache_path)
        self.window_size = window_size
        self.max_ratings = max_ratings
        self.min_count = min_count
        self.num_negative = num_negative

        if rebuild_cache or not os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)
            self._build_cache(cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
            with open(os.path.join(cache_path, "movie.pkl"), "rb") as fi:
                self.movies = pickle.load(fi)
            with open(os.path.join(cache_path, "vocab.pkl"), "rb") as fi:
                movie_vocab, genres_vocab, country_vocab = pickle.load(fi)
            with open(os.path.join(cache_path, "freqs.pkl"), "rb") as fi:
                self.freqs = pickle.load(fi)
            self.movie_vocab = movie_vocab
            self.genres_vocab = genres_vocab
            self.country_vocab = country_vocab
            self.all_movies = np.array(range(len(self.movie_vocab)))
    
    def _load_movies(self):
        print("loading movies")
        rated_movies = set()
        for ratings in self.ratings.values():
            rated_movies.update(ratings["likes"] + ratings["dislikes"])

        with open(self.movie_file, encoding="utf-8") as fi:
            movies = [json.loads(line) for line in fi]
            self.movies = {}
            for movie in movies:
                movie_id = movie["id"]
                if movie_id in rated_movies:
                    genres = movie.get("genres", None)
                    if genres:
                        genres = genres[0]
                    else:
                        genres = None
                    country = movie.get("producing_countries", None)
                    if country:
                        country = country[0]
                    else:
                        country = None
                    self.movies[movie_id] = {"genres": genres, "country": country}

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            central_id, context = pickle.loads(txn.get(struct.pack(">I", index)))
            negatives = np.random.choice(self.all_movies, size=len(context)*self.num_negative, p=self.freqs)
            labels = [1] * len(context) + [0] * len(negatives)
            context = [self.movie_vocab[movie_id] for movie_id in context]
            context.extend(negatives)
            idxes = list(range(len(context)))
            random.shuffle(idxes)
            context = [context[idx] for idx in idxes]
            labels = [labels[idx] for idx in idxes]

            central_idx = self.movie_vocab[central_id]
            genres = self.movies[central_id]["genres"]
            genres_oov = len(self.genres_vocab)
            genres = self.genres_vocab.get(genres, genres_oov)
            country = self.movies[central_id]["country"]
            country_oov = len(self.country_vocab)
            country = self.country_vocab.get(country, country_oov)
            return central_idx, genres, country, np.array(context), np.array(labels)

    def build_vocabularies(self):
        counts = Counter(chain.from_iterable(value["likes"] for value in self.ratings.values()))
        items, freqs = zip(*[(movie_id, count) for movie_id, count in counts.items() if count >= self.min_count])
        self.movie_vocab = {movie_id: i for i, movie_id in enumerate(items)}
        freqs = np.array(freqs) ** 0.75
        freqs = freqs / freqs.sum()
        self.freqs = freqs
    
        genres_counts = Counter([movie["genres"] for movie in self.movies.values()])
        genres = [genres for genres, count in genres_counts.items() if count >= self.min_count]
        self.genres_vocab = {genres_id: i for i, genres_id in enumerate(genres)}

        country_counts = Counter([movie["country"] for movie in self.movies.values()])
        countrys = [country for country, count in country_counts.items() if count >= self.min_count]
        self.country_vocab = {country: i for i, country in enumerate(countrys)}

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        with open(os.path.join(self.cache_path, "vocab.pkl"), "wb") as fo:
            pickle.dump((self.movie_vocab, self.genres_vocab, self.country_vocab), fo)
        with open(os.path.join(self.cache_path, "freqs.pkl"), "wb") as fo:
                pickle.dump(self.freqs, fo)

    def _yield_buffer(self, buffer_size=1000):
        buffer = list()
        item_idx = 0
        pbar = tqdm(self.ratings.values(), mininterval=1, smoothing=0.1)
        pbar.set_description("processing ratings")
        for rating in pbar:
            likes = rating["likes"]
            likes = [movie_id for movie_id in likes if movie_id in self.movie_vocab]
            for i in range(self.window_size, len(likes)-self.window_size):
                central_id = likes[i]
                context = likes[i-self.window_size:i] + likes[i+1:i+self.window_size]
                buffer.append((struct.pack(">I", item_idx), pickle.dumps((central_id, context))))
                item_idx += 1
                if len(buffer) == buffer_size:
                    yield buffer
                    buffer.clear()
        yield buffer


class MovieRecallDataset(MovieDataset):
    def __init__(self, movie_file, rating_file, cache_path="data/.movie_recall", rebuild_cache=False, max_ratings=100, 
                min_ratings=5, num_negative=3, history_size=20):
        super().__init__(movie_file, rating_file, None, None)
        self.history_size = history_size
        
    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            history, positive = pickle.loads(txn.get(struct.pack(">I", index)))
            history = [self.movie_vocab[movie_id] + 1 for movie_id in history]
            history = history + [0] * max(0, self.history_size - len(history))
            history = np.array(history)
            positive = self.movie_vocab[positive] + 1
            return history, positive

    def _yield_buffer(self, buffer_size=1000):
        buffer = list()
        item_idx = 0
        pbar = tqdm(self.ratings.values(), mininterval=1, smoothing=0.1)
        pbar.set_description("processing ratings")
        for rating in pbar:
            likes = rating["likes"]
            
            for i in range(self.min_ratings-1, len(likes)):
                start = max(0, i-self.history_size)
                history = likes[start:i]
                buffer.append((struct.pack(">I", item_idx), pickle.dumps((history, likes[i]))))
                item_idx += 1
                if len(buffer) == buffer_size:
                    yield buffer
                    buffer.clear()
        yield buffer


class MovieRankdingDataset(MovieDataset):
    def __init__(self, movie_file, rating_file, cache_path="data/.movie_rank", rebuild_cache=False, max_ratings=100, 
                min_ratings=5, num_negative=2, history_size=20):
        super().__init__(movie_file, rating_file, max_celebrities, max_tags)
        self.history_size = history_size
        if rebuild_cache or not os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)
            self._build_cache(cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
            with open(os.path.join(cache_path, "movie.pkl"), "rb") as fi:
                self.movies = pickle.load(fi)
            with open(os.path.join(cache_path, "vocab.pkl"), "rb") as fi:
                movie_vocab, celebrity_vocab, tag_vocab = pickle.load(fi)
            with open(os.path.join(cache_path, "freqs.pkl"), "rb") as fi:
                self.freqs = pickle.load(fi)
            self.movie_vocab = movie_vocab
            self.all_movies = np.array(range(len(self.movie_vocab)))
            self.celebrity_vocab = celebrity_vocab
            self.tag_vocab = tag_vocab
            self.celebrity_oov = len(self.celebrity_vocab) + 1
            self.tag_oov = len(self.tag_vocab) + 1

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            history, candidates, labels = pickle.loads(txn.get(struct.pack(">I", index)))
            history_size = len(history)
            history, history_seq_lens = zip(*[self._convert_movie(movie_id) for movie_id in history])
            history = list(history)
            history_seq_lens = list(history_seq_lens)
            max_seq_len = self.max_celebrities + self.max_tags
            history += [[0] * max_seq_len] * (self.history_size - len(history))
            history_seq_lens += [0] * (self.history_size - len(history_seq_lens))
            candidates, candidate_seq_lens = zip(*[self._convert_movie(movie_id) for movie_id in candidates])
            candidates = list(candidates)
            candidate_seq_lens = list(candidate_seq_lens)
            candidates += [[0] * max_seq_len] * (1 + self.num_negative - len(candidates))
            candidate_seq_lens += [0] * (1 + self.num_negative - len(candidate_seq_lens))
            labels += [0] * (1 + self.num_negative - len(labels))
            return np.array(labels), np.array(history), history_size, np.array(history_seq_lens), np.array(candidates), np.array(candidate_seq_lens)

    def _convert_movie(self, movie_id):
        celebrities = [self.celebrity_vocab.get(celebrity, self.celebrity_oov) for celebrity in self.movies[movie_id]["celebrities"]]
        tags = [self.tag_vocab.get(tag, self.tag_oov) for tag in self.movies[movie_id]["tags"]]
        movie = celebrities[:self.max_celebrities] + tags
        max_seq_len = self.max_celebrities + self.max_tags
        seq_len = min(len(movie), max_seq_len)
        movie = movie[:max_seq_len] + [0] * max(0, max_seq_len-len(movie))
        return movie, seq_len
    
    def build_vocabularies(self):
        self.movie_vocab = {id: i for i, id in enumerate(self.movies.keys())}
        self.freqs = None

        celebrity_counts = Counter(chain.from_iterable([movie["celebrities"] for movie in self.movies.values()]))
        celebrities = [celebrity for celebrity, count in celebrity_counts.items() if count >= self.min_count]
        self.celebrity_vocab = {celebrity_id: i + 1 for i, celebrity_id in enumerate(celebrities)}
        celebrity_vocab_size = len(self.celebrity_vocab) + 2
        tag_counts = Counter(chain.from_iterable([movie["tags"] for movie in self.movies.values()]))
        tags = [tag for tag, count in tag_counts.items() if count >= self.min_count]
        self.tag_vocab = {tag: i + celebrity_vocab_size for i, tag in enumerate(tags)}

    def _yield_buffer(self, buffer_size=1000):
        buffer = list()
        item_idx = 0
        pbar = tqdm(self.ratings.values(), mininterval=1, smoothing=0.1)
        pbar.set_description("processing ratings")
        for rating in pbar:
            likes = rating["likes"]
            dislikes = rating["dislikes"]
            if len(dislikes) < 2:
                continue
            more_negatives = len(likes) * self.num_negative - len(dislikes)
            if more_negatives > 0:
                dislikes.extend(np.random.choice(dislikes, more_negatives))
            for i in range(self.min_ratings-1, len(likes)):
                start = max(0, i - self.history_size)
                history = likes[start:i]
                positive = likes[i]
                negatives = dislikes[(i-self.min_ratings+1)*self.num_negative:(i-self.min_ratings+2)*self.num_negative]
                candidates = [positive] + negatives
                labels = [1] + [0] * len(negatives)
                idxes = list(range(len(candidates)))
                random.shuffle(idxes)
                candidates = [candidates[idx] for idx in idxes]
                labels = [labels[idx] for idx in idxes]
                buffer.append((struct.pack(">I", item_idx), pickle.dumps((history, candidates, labels))))
                item_idx += 1
                if len(buffer) == buffer_size:
                    yield buffer
                    buffer.clear()
        yield buffer
