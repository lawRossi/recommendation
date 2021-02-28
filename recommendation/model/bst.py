from .nrms import DocEncoder
import torch.nn as nn
import torch
import numpy as np


class BSTModel(nn.Module):
    def __init__(self, vocab_size, emb_dims, num_doc_encoder_heads, attention_hidden_dim=100, 
            num_transformer_heads=10, history_size=20, weights=None, discrete_vocab_sizes=None,
            discrete_emb_dims=20, real_value_dims=0, dropout=0.2, device="cpu"):
        super().__init__()
        self.doc_encoder = DocEncoder(vocab_size, emb_dims, num_doc_encoder_heads, weights, attention_hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(emb_dims, num_transformer_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, 1)
        self.real_value_dims = real_value_dims
        hidden_size = (history_size+1) * emb_dims
        if discrete_vocab_sizes:
            self.discrete_embeddings = nn.ModuleList([nn.Embedding(size, discrete_emb_dims) for size in discrete_vocab_sizes])
            hidden_size += len(discrete_vocab_sizes) * discrete_emb_dims
        hidden_size += real_value_dims
        self.hidden_layers = nn.Sequential(nn.Linear(hidden_size, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.device = device
    
    def forward(self, click_history, history_seq_lens, candidates, candidate_seq_lens, discretes=None, real_values=None):
        batch_size, history_size, seq_len = click_history.shape
        reshaped_history = click_history.reshape(-1, seq_len)
        history_seq_lens = history_seq_lens.reshape(-1, 1)
        masks = self._compute_masks(seq_len, history_seq_lens)
        encoded_history = self.doc_encoder(reshaped_history, masks)
        encoded_history = encoded_history.reshape(batch_size, history_size, -1)

        masks = self._compute_masks(seq_len, candidate_seq_lens)
        encoded_candidates = self.doc_encoder(candidates, masks)
        encoded_candidates = encoded_candidates.unsqueeze(1)

        concated = torch.cat((encoded_history, encoded_candidates), dim=1)
        transformed = self.transformer(concated)
        flattend = transformed.reshape(batch_size, -1)

        discrete_embs = []
        if discretes is not None:
            for discrete, embedding in zip(discretes, self.discrete_embeddings):
                discrete_embs.append(embedding(discrete))

        to_concat = [flattend] + discrete_embs
        if real_values is not None:
            to_concat.append(real_values)
        concated = torch.cat(to_concat, dim=1)
        hidden = self.hidden_layers(concated)
        output = self.output(hidden)
        return output

    def _compute_masks(self, max_seq_len, seq_lens):
        if len(seq_lens.shape) == 2:
            masks = torch.arange(max_seq_len, device=self.device) >= seq_lens
        else:
            masks = torch.arange(max_seq_len, device=self.device) >= seq_lens[:, None]
        return masks


if __name__ == "__main__":
    model = BSTModel(50, 50, 5, discrete_vocab_sizes=[5, 5], history_size=3)

    x1 = np.array([[[1, 2, 3, 0], [1, 2, 4, 0], [0, 0, 0, 0]]])
    seq_lens1 = np.array([[3, 2, 1]])

    x2 = np.array([[1, 2, 4, 0]])
    seq_lens2 = np.array([3])

    x1 = torch.tensor(x1, dtype=torch.long)
    seq_lens1 = torch.tensor(seq_lens1, dtype=torch.long)

    x2 = torch.tensor(x2, dtype=torch.long)
    seq_lens2 = torch.tensor(seq_lens2, dtype=torch.long)

    discretes = [torch.tensor([2], dtype=torch.long), torch.tensor([1], dtype=torch.long)]

    sigmoids = model(x1, seq_lens1, x2, seq_lens2, discretes)
    print(sigmoids)
