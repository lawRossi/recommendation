import jieba
import torch
import json
import os.path


class NRMSModel:
    def __init__(self, model_path, device="cpu"):
        vocab_file = os.path.join(model_path, "vocab.json")
        with open(vocab_file, encoding="utf-8") as fi:
            word_dictionary = json.load(fi)
        self.word_dictionary = word_dictionary
        self.OOV_IDX = len(self.word_dictionary) + 1
        self.device = torch.device(device)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.device = device
        self.model.eval()

    def encode_docs(self, docs, max_seq_len=15):
        doc_tokens = [jieba.lcut(doc) for doc in docs]
        token_idxes_list = self.tokens2idxes(doc_tokens, max_seq_len)
        seq_lens = [min(len(tokens), max_seq_len) for tokens in doc_tokens]
        tokens_tensor = torch.tensor(token_idxes_list, dtype=torch.long, device=self.device)
        seq_lens = torch.tensor(seq_lens, device=self.device)
        encoded = self.model.encode_docs(tokens_tensor, seq_lens)
        return encoded.cpu().detach().numpy()

    def encode_users(self, history_list, max_seq_len=15, history_size=30):
        history_clicks = []
        click_seq_lens = []
        history_sizes = []
        for history in history_list:
            doc_tokens = [jieba.lcut(doc) for doc in history]
            token_idxes_list = self.tokens2idxes(doc_tokens, max_seq_len)
            token_idxes_list += [[0] * max_seq_len] * (history_size - len(history))
            seq_lens = [min(len(tokens), max_seq_len) for tokens in doc_tokens] + [0] * (history_size - len(history))
            history_clicks.append(token_idxes_list)
            click_seq_lens.append(seq_lens)
            history_sizes.append(len(history))
        history_clicks = torch.tensor(history_clicks, dtype=torch.long)
        click_seq_lens = torch.tensor(click_seq_lens)
        history_sizes = torch.tensor(history_sizes)
        encoded_users = self.model.encode_users(history_clicks, click_seq_lens, history_sizes)
        return encoded_users.cpu().detach().numpy()

    def tokens2idxes(self, tokens_list, max_seq_len):
        dictionary = self.word_dictionary
        token_idxes_list = []
        for tokens in tokens_list:
            token_idxes = [dictionary[token] if token in dictionary else self.OOV_IDX for token in tokens]
            token_idxes = token_idxes[:max_seq_len] + [0] * (max_seq_len-len(token_idxes))
            token_idxes_list.append(token_idxes)
        return token_idxes_list


class BSTModel:
    def __init__(self, model_path, vocab_file, attr_vocab_file, device="cpu"):
        with open(vocab_file, encoding="utf-8") as fi:
            word_dictionary = json.load(fi)
        with open(attr_vocab_file, encoding="utf-8") as fi:
            attr_vocabs = json.load(fi)
        self.word_dictionary = word_dictionary
        self.attr_vocabs = attr_vocabs
        self.OOV_IDX = len(self.word_dictionary) + 1
        self.device = torch.device(device)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.device = self.device
        self.model.eval()

    def tokens2idxes(self, tokens_list, max_seq_len):
        dictionary = self.word_dictionary
        token_idxes_list = []
        for tokens in tokens_list:
            token_idxes = [dictionary[token] if token in dictionary else self.OOV_IDX for token in tokens]
            token_idxes = token_idxes[:max_seq_len] + [0] * (max_seq_len-len(token_idxes))
            token_idxes_list.append(token_idxes)
        return token_idxes_list
    
    def predict(self, history, candidates, attrs=None, max_seq_len=15, history_size=30):
        doc_tokens = [jieba.lcut(doc) for doc in history]
        token_idxes_list = self.tokens2idxes(doc_tokens, max_seq_len)
        token_idxes_list += [[0] * max_seq_len] * (history_size - len(history))
        seq_lens = [min(len(tokens), max_seq_len) for tokens in doc_tokens] + [0] * (history_size - len(history))
        
        history_clicks = [token_idxes_list] * len(candidates)
        history_clicks = torch.tensor(history_clicks, dtyp=torch.long, device=self.device)
        click_seq_lens = [seq_lens] * len(candidates)
        click_seq_lens = torch.tensor(click_seq_lens, device=self.device)

        doc_tokens = [jieba.lcut(token) for token in candidates]
        token_idxes_list = self.tokens2idxes(doc_tokens, max_seq_len)
        candidate_seq_lens = [min(len(tokens), max_seq_len) for tokens in doc_tokens]
        candidates = torch.tensor(token_idxes_list, dtype=torch.long, device=self.device)
        candidate_seq_lens = torch.tensor(candidate_seq_lens, device=self.device)

        if attrs:
            discretes = []
            for attr_name, vocab in self.attr_dictionary.items():
                attr_value = attrs.get(attr_name)
                attr_idx = vocab.get(attr_value, len(vocab))
                discrete = torch.tensor([attr_idx] * len(candidates), dtype=torch.long, device=self.device)
                discretes.append(discrete)
        else:
            discretes = None
        
        scores = self.mode(history_clicks, click_seq_lens, candidates, candidate_seq_lens, discretes)
        return scores.cpu().detach().numpy()


class YoutubeNetModel:
    def __init__(self, model_dir, max_history=30, device="cpu"):
        model_path = os.path.join(model_dir, "youtube_net.pt")
        self.model = torch.load(model_path, map_location=device)
        vocab_file = os.path.join(model_dir, "vocab.json")
        with open(vocab_file, encoding="utf-8") as fi:
            self.vocab = json.load(fi)
        attr_vocab_file = os.path.join(model_dir, "attr_vocabs.json")
        if os.path.exists(attr_vocab_file):
            with open(attr_vocab_file, encoding="utf-8") as fi:
                self.attr_vocabs = json.load(fi)
        else:
            self.attr_vocabs = None
        self.max_history = max_history
        self.device = device
    
    def encode_user(self, click_history, discrete_attrs):
        click_history = [self.vocab[item_id] for item_id in click_history if item_id in self.vocab]
        click_history = click_history[-self.max_history:]
        history_tensor = torch.tensor([click_history], dtype=torch.long, device=self.device)
        if discrete_attrs:
            discrete_tensors = []
            for attr_name, attr_value in discrete_attrs.items():
                vocab = self.attr_vocabs[attr_name]
                attr_idx = vocab.get(attr_value, len(vocab))
                discrete_tensors.append(torch.tensor([attr_idx], dtype=torch.long))
        else:
            discrete_tensors = None
        encodings = self.model(history_tensor, discrete_features=discrete_tensors)
        return encodings.cpu().detach().numpy()
