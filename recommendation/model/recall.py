import torch
import torch.nn as nn
import numpy as np
import numpy as np
from .nrms import AdditiveAttention


class SampledSoftmaxLoss(nn.Module):
    def __init__(self, vocab_size, emb_dims, num_negatives=5, weights=None, padding=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embdding = nn.Embedding(vocab_size, emb_dims)
        self.num_negatives = num_negatives
        if weights is not None:
            weights = np.power(weights, 0.75)
            weights = weights / weights.sum()
            self.weights = torch.tensor(weights, dtype=torch.float)
        else:
            self.weights = None
        self.padding = padding
    
    def forward(self, hidden, positives):
        if len(positives.shape) == 1:
            batch_size, context_size = positives.shape[0], 1
        else:
            batch_size, context_size = positives.shape
        if self.weights is not None:
            negatives = torch.multinomial(self.weights, batch_size * context_size * self.num_negatives, replacement=False).view(batch_size, -1)
        else:
            min = 0 if not self.padding else 1
            negatives = torch.FloatTensor(batch_size, context_size * self.num_negatives).uniform_(min, self.vocab_size - 1).long()
        negatives = negatives.to(positives.device)
        embedded_positives = self.embdding(positives)
        if len(embedded_positives.shape) == 2:
            embedded_positives = embedded_positives.unsqueeze(1)
        embedded_negaitves = self.embdding(negatives).neg()
        if len(hidden.shape) == 2:
            hidden = hidden.unsqueeze(2)
        p_loss = torch.bmm(embedded_positives, hidden).squeeze(-1).sigmoid().log().mean(dim=1)
        n_loss = torch.bmm(embedded_negaitves, hidden).squeeze().sigmoid().log().view(-1, context_size, self.num_negatives).sum(dim=2).mean(dim=1)
        return -(p_loss + n_loss).mean()


class YoutubeNetModel(nn.Module):
    def __init__(self, num_items, item_emb_dims=100, discrete_vocab_sizes=None, discrete_emb_dims=20, num_real_values=0,
            num_negatives=5, weights=None):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, item_emb_dims)
        feature_emb_dims = item_emb_dims
        if discrete_vocab_sizes:
            self.discrete_embeddings = nn.ModuleList([nn.Embedding(vocab_size, discrete_emb_dims) for vocab_size in discrete_vocab_sizes])
            feature_emb_dims += len(discrete_vocab_sizes) * discrete_emb_dims
        feature_emb_dims += num_real_values
        hidden_dims = [item_emb_dims * 4, item_emb_dims*2, item_emb_dims]
        self.hidden_layers = nn.Sequential(nn.Linear(feature_emb_dims, hidden_dims[0]), nn.ReLU(), nn.Linear(hidden_dims[0], hidden_dims[1]), 
            nn.ReLU(), nn.Linear(hidden_dims[1], hidden_dims[2]), nn.ReLU())
        self.loss = SampledSoftmaxLoss(num_items, item_emb_dims, num_negatives=num_negatives, weights=weights)

    def forward(self, history, positives, discrete_features=None, real_value_features=None, training=True):
        """[summary]

        Args:
            history (tensor): tensor of shape (batch_size, history_size)
            discrete_features (list, optional): a list of tensor of shape (batch_size, ). Defaults to None.
            real_value_features (list, optional): tensor of shape (batch_size, num_features). Defaults to None.
        """
        embedded_history = self.item_embedding(history).mean(dim=1)  # batch_size x emb_dims
        features = [embedded_history]
        if discrete_features is not None:
            embedded_discrete_features = []
            for i, item in enumerate(discrete_features):
                embedded_discrete_features.append(self.discrete_embeddings[i](item))
                if len(embedded_discrete_features[i].shape) == 3:
                    embedded_discrete_features[i] = embedded_discrete_features[i].mean(dim=1)
            features.extend(embedded_discrete_features)
        if real_value_features is not None:
            features.append(real_value_features)
        embedded_features = torch.cat(features, dim=-1)
        hidden = self.hidden_layers(embedded_features)
        if training:
            loss = self.loss(hidden, positives)
            return loss
        else:
            return hidden


class MhaRecallModel(nn.Module):
    def __init__(self, num_items, item_emb_dims, num_heads, discrete_vocab_sizes=None, discrete_emb_dims=20, 
            num_real_values=0, num_negatives=5, weights=None, additive_atten_dims=100, dropout=0.2, device="cpu"):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, item_emb_dims)
        self.mha = nn.MultiheadAttention(item_emb_dims, num_heads, dropout=dropout)
        self.addtive = AdditiveAttention(item_emb_dims, additive_atten_dims)

        if discrete_vocab_sizes or num_real_values != 0:
            feature_emb_dims = item_emb_dims
            if discrete_vocab_sizes:
                self.discrete_embeddings = nn.ModuleList([nn.Embedding(vocab_size, discrete_emb_dims) for vocab_size in discrete_vocab_sizes])
                feature_emb_dims += len(discrete_vocab_sizes) * discrete_emb_dims
            feature_emb_dims += num_real_values
            hidden_dims = [item_emb_dims * 4, item_emb_dims*2, item_emb_dims]
            self.hidden_layers = nn.Sequential(nn.Linear(feature_emb_dims, hidden_dims[0]), nn.ReLU(), nn.Linear(hidden_dims[0], hidden_dims[1]), 
                nn.ReLU(), nn.Linear(hidden_dims[1], hidden_dims[2]), nn.ReLU())

        self.loss = SampledSoftmaxLoss(num_items, item_emb_dims, num_negatives=num_negatives, weights=weights)
        self.device = torch.device(device)

    def forward(self, history, history_sizes, positives, discrete_features=None, real_value_features=None, training=True):
        """[summary]

        Args:
            history (tensor): tensor of shape (batch_size, history_size)
            discrete_features (list, optional): a list of tensor of shape (batch_size, ). Defaults to None.
            real_value_features (list, optional): tensor of shape (batch_size, num_features). Defaults to None.
        """
        max_history_size = history.shape[1]
        embedded_history = self.item_embedding(history)  # batch_size x max_history x emb_dims
        permuted = embedded_history.permute(1, 0, 2)
        mha_attended, _ = self.mha(permuted, permuted, permuted)
        mha_attended = mha_attended.permute(1, 0, 2)
        masks = self._compute_mask(max_history_size, history_sizes)
        encoded_history = self.addtive(mha_attended, masks)

        features = [encoded_history]
        if discrete_features:
            embedded_discrete_features = []
            for i, item in enumerate(discrete_features):
                embedded_discrete_features.append(self.discrete_embeddings[i](item))
                if len(embedded_discrete_features[i].shape) == 3:
                    embedded_discrete_features[i] = embedded_discrete_features[i].mean(dim=1)
            features.extend(embedded_discrete_features)
        if real_value_features is not None:
            features.append(real_value_features)
        if discrete_features is not None or real_value_features is not None:
            embedded_features = torch.cat(features, dim=-1)
            hidden = self.hidden_layers(embedded_features)
        else:
            hidden = encoded_history
        if training:
            loss = self.loss(hidden, positives)
            return loss
        else:
            return hidden

    def _compute_mask(self, max_seq_len, seq_lens):
        mask = torch.arange(max_seq_len, device=self.device) >= seq_lens[:, None]
        return mask


if __name__ == "__main__":
    # model = YoutubeNetModel(10, 20, [5, 5], 10, num_real_values=3)
    # history = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
    # discrete_features = [torch.tensor([[1, 2], [2, 3]], dtype=torch.long), torch.tensor([2, 3], dtype=torch.long)]
    # real_value_features = torch.tensor([[0.1, 0.2, 0.1], [0.2, 0.4, 0.3]])
    # positives = torch.tensor([4, 5], dtype=torch.long)
    # loss = hidden = model(history, positives, discrete_features, real_value_features)
    # print(loss)


    model = MhaRecallModel(10, 20, 4, [5, 5], num_real_values=3)
    history = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
    history_sizes = torch.tensor([3, 3])
    discrete_features = [torch.tensor([[1, 2], [2, 3]], dtype=torch.long), torch.tensor([2, 3], dtype=torch.long)]
    real_value_features = torch.tensor([[0.1, 0.2, 0.1], [0.2, 0.4, 0.3]])
    positives = torch.tensor([4, 5], dtype=torch.long)
    loss = model(history, history_sizes, positives, discrete_features, real_value_features)
    print(loss)
