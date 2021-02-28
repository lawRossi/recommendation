import torch 
import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self, in_dim, attention_hidden_dim=100):
        super().__init__()
        self.in_dim = in_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.projection = nn.Sequential(nn.Linear(in_dim, attention_hidden_dim), nn.Tanh())
        self.query = nn.Linear(attention_hidden_dim, 1, bias=False)
    
    def forward(self, x, masks=None):
        weights = self.query(self.projection(x)).squeeze(-1)
        if masks is not None:
            weights = weights.masked_fill(masks, -1e9)
        weights = torch.softmax(weights, dim=-1)
        output = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return output


class DocEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, embedding_weights=None, attention_hidden_dim=100, dropout=0.2):
        super().__init__()
        if embedding_weights is None:
            self.embbedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        else:
            weights = torch.tensor(embedding_weights, dtype=torch.float)
            self.embbedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        self.mha = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.attention = AdditiveAttention(emb_dim, attention_hidden_dim)

    def forward(self, x, masks=None):
        """

        Args:
            x : tensor with shape (batch_size, seq_len)
        """
        embedded = self.dropout(self.embbedding(x))
        permuted = embedded.permute(1, 0, 2)
        attended, _ = self.mha(permuted, permuted, permuted)
        attended = attended.permute(1, 0, 2)
        output = self.attention(attended, masks)
        return output


class UserEncoder(nn.Module):
    def __init__(self, emb_dim, num_heads, attention_hidden_dim=100, dropout=0.2,):
        super().__init__()
        self.mha = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout)
        self.attention = AdditiveAttention(emb_dim, attention_hidden_dim)
    
    def forward(self, encoded_docs, masks=None):
        encoded_docs = encoded_docs.permute(1, 0, 2)
        attended_docs, _ = self.mha(encoded_docs, encoded_docs, encoded_docs)
        attended_docs = attended_docs.permute(1, 0, 2)
        output = self.attention(attended_docs, masks)
        return output

class NRMS(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, attention_hidden_dim=100, embedding_weights=None, dropout=0.2, device="cpu"):
        super().__init__()
        self.doc_encoder = DocEncoder(vocab_size, emb_dim, num_heads, embedding_weights, attention_hidden_dim, dropout)
        self.user_encoder = UserEncoder(emb_dim, num_heads, attention_hidden_dim, dropout)
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, click_history, history_lens, history_seq_lens, candidates, candidate_seq_lens):
        """

        Args:
            click_history : tensor with shape (batch_size, history_size, seq_len)
            candidates : tensor with shape (batch_size, num_candidates, seq_len)
        """
        batch_size, history_size, seq_len = click_history.shape
        num_candidates = candidates.shape[1]
        reshaped_history = click_history.reshape(-1, seq_len)
        history_seq_lens = history_seq_lens.reshape(-1, 1)
        masks = self._compute_mask(seq_len, history_seq_lens)
        encoded_history = self.doc_encoder(reshaped_history, masks)
        encoded_history = encoded_history.reshape(batch_size, history_size, -1)
        masks = self._compute_mask(history_size, history_lens)
        encoded_users = self.user_encoder(encoded_history, masks)

        reshaped_candidates = candidates.reshape(-1, seq_len)
        candidate_seq_lens = candidate_seq_lens.reshape(-1, 1)
        masks = self._compute_mask(seq_len, candidate_seq_lens)
        encoded_candidates = self.doc_encoder(reshaped_candidates, masks)
        encoded_candidates = encoded_candidates.reshape(batch_size, num_candidates, -1)
        logits = torch.bmm(encoded_users.unsqueeze(1), encoded_candidates.permute(0, 2, 1)).squeeze(1)
        return torch.sigmoid(logits)

    def _compute_mask(self, max_seq_len, seq_lens):
        if len(seq_lens.shape) == 2:
            mask = torch.arange(max_seq_len, device=self.device) >= seq_lens
        else:
            mask = torch.arange(max_seq_len, device=self.device) >= seq_lens[:, None]
        return mask

    def encode_docs(self, docs, seq_lens=None):
        if seq_lens is not None:
            _, max_seq_len = docs.shape
            masks = self._compute_mask(max_seq_len, seq_lens)
        else:
            masks = None
        return self.doc_encoder(docs, masks)
    
    def encode_user(self, history_clicks, click_seq_lens, history_sizes):
        batch_size, history_size, seq_len = history_clicks.shape
        reshaped_clicks = history_clicks.reshpae(-1, seq_len)
        click_seq_lens = click_seq_lens.reshape(-1, seq_len)
        mask = self._compute_mask(seq_len, click_seq_lens)
        encoded_clicks = self.doc_encoder(reshaped_clicks, mask)
        encoded_history = encoded_clicks.reshape(batch_size, history_size, -1)
        mask = self._compute_mask(history_size, history_sizes)
        encoded_users = self.user_encoder(encoded_history, mask)
        return encoded_users
