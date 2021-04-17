import torch
import torch.nn as nn
from .nrms import DocEncoder


class SampledSoftmaxLoss(nn.Module):
    def __init__(self, vocab_size, emb_dims, num_negatives=5, weights=None, padding=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embdding = nn.Embedding(vocab_size, emb_dims)
        init_range = 0.5 / emb_dims
        self.embdding.weight.data.uniform_(-init_range, init_range)
        self.num_negatives = num_negatives
        if weights is not None:
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
            num_negatives=5, weights=None, droput=0.2):
        super().__init__()
        self.num_items = num_items
        self.item_emb_dims = item_emb_dims
        self.item_embedding = nn.Embedding(num_items, item_emb_dims, padding_idx=0)
        feature_emb_dims = item_emb_dims
        if discrete_vocab_sizes:
            self.discrete_embeddings = nn.ModuleList([nn.Embedding(vocab_size, discrete_emb_dims) for vocab_size in discrete_vocab_sizes])
            feature_emb_dims += len(discrete_vocab_sizes) * discrete_emb_dims
        feature_emb_dims += num_real_values
        hidden_dims = [item_emb_dims * 4, item_emb_dims*2, item_emb_dims]
        self.hidden_layers = nn.Sequential(nn.Linear(feature_emb_dims, hidden_dims[0]), nn.ReLU(), nn.Linear(hidden_dims[0], hidden_dims[1]), 
            nn.ReLU(), nn.Linear(hidden_dims[1], hidden_dims[2]), nn.ReLU())
        self.dropout = nn.Dropout(droput)
        self.loss = SampledSoftmaxLoss(num_items, item_emb_dims, num_negatives=num_negatives, weights=weights)

    def forward(self, click_history, positives=None, discrete_features=None, real_value_features=None):
        """[summary]

        Args:
            history (tensor): tensor of shape (batch_size, history_size)
            discrete_features (list, optional): a list of tensor of shape (batch_size, ). 
            real_value_features (list, optional): tensor of shape (batch_size, num_features).
        """
        embedded_history = self.item_embedding(click_history).mean(dim=1)  # batch_size x emb_dims
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
        embedded_features = self.dropout(torch.cat(features, dim=-1))
        hidden = self.hidden_layers(embedded_features)
        if positives is not None:
            loss = self.loss(hidden, positives)
            return loss
        else:
            return hidden


class YoutubeNetSeqModel(nn.Module):
    def __init__(self, vocab_size, item_emb_dims, discrete_vocab_sizes=None, discrete_emb_dims=20, num_real_values=0,
            embedding_weights=None, droput=0.2):
        super().__init__()
        self.item_encoder = DocEncoder(vocab_size, item_emb_dims, num_heads=10, embedding_weights=embedding_weights)
        feature_emb_dims = item_emb_dims
        if discrete_vocab_sizes:
            self.discrete_embeddings = nn.ModuleList([nn.Embedding(vocab_size, discrete_emb_dims) for vocab_size in discrete_vocab_sizes])
            feature_emb_dims += len(discrete_vocab_sizes) * discrete_emb_dims
        feature_emb_dims += num_real_values
        hidden_dims = [item_emb_dims * 4, item_emb_dims*2, item_emb_dims]
        self.droput = nn.Dropout(droput)
        self.hidden_layers = nn.Sequential(nn.Linear(feature_emb_dims, hidden_dims[0]), nn.ReLU(), nn.Linear(hidden_dims[0], hidden_dims[1]), 
            nn.ReLU(), nn.Linear(hidden_dims[1], hidden_dims[2]), nn.ReLU())
        self.dropout = nn.Dropout(droput)
        self.loss = nn.BCELoss()

    def forward(self, click_history, candidates=None, labels=None, discrete_features=None, real_value_features=None):
        """[summary]

        Args:
            history (tensor): tensor of shape (batch_size, history_size, sequence_len)
            candidates (tensor): tensor of shape (batch_size, num_candidates, sequence_len)
            labels (tensor): tensor of shape (batch_size, num_candidates)
            discrete_features (list, optional): a list of tensor of shape (batch_size, ). Defaults to None.
            real_value_features (list, optional): tensor of shape (batch_size, num_features). Defaults to None.
        """
        batch_size, history_size, seq_len = click_history.shape
        reshaped_history = click_history.reshape(-1, seq_len)
        encoded_history = self.item_encoder(reshaped_history)
        encoded_history = encoded_history.reshape(batch_size, history_size, -1)
        encoded_history = encoded_history.mean(dim=1)

        features = [encoded_history]
        if discrete_features is not None:
            embedded_discrete_features = []
            for i, item in enumerate(discrete_features):
                embedded_discrete_features.append(self.discrete_embeddings[i](item))
                if len(embedded_discrete_features[i].shape) == 3:
                    embedded_discrete_features[i] = embedded_discrete_features[i].mean(dim=1)
            features.extend(embedded_discrete_features)
        if real_value_features is not None:
            features.append(real_value_features)
        embedded_features = self.dropout(torch.cat(features, dim=-1))
        hidden = self.hidden_layers(embedded_features)

        if candidates is not None:
            _, num_candidates, _ = candidates.shape
            reshaped_candidates = candidates.reshape(-1, seq_len)
            encoded_candidates = self.item_encoder(reshaped_candidates)
            encoded_candidates = encoded_candidates.reshape(batch_size, num_candidates, -1)
            print(encoded_candidates.shape)
            logits = torch.bmm(hidden.unsqueeze(1), encoded_candidates.permute(0, 2, 1)).squeeze(1)
            probas = torch.sigmoid(logits)
            print(probas)
            loss = self.loss(probas, labels)
            return loss
        else:
            return hidden


if __name__ == "__main__":
    # model = YoutubeNetModel(10, 20, [5, 5], 10, num_real_values=3)
    # history = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
    # discrete_features = [torch.tensor([[1, 2], [2, 3]], dtype=torch.long), torch.tensor([2, 3], dtype=torch.long)]
    # real_value_features = torch.tensor([[0.1, 0.2, 0.1], [0.2, 0.4, 0.3]])
    # positives = torch.tensor([4, 5], dtype=torch.long)
    # loss = model(history, positives, discrete_features, real_value_features)
    # print(loss)

    model = YoutubeNetSeqModel(10, 20, [5, 5], 10, num_real_values=3)
    history = torch.tensor([[[1, 2, 3], [1, 2, 3]], [[2, 3, 4], [1, 2, 3]]], dtype=torch.long)
    discrete_features = [torch.tensor([[1, 2], [2, 3]], dtype=torch.long), torch.tensor([2, 3], dtype=torch.long)]
    real_value_features = torch.tensor([[0.1, 0.2, 0.1], [0.2, 0.4, 0.3]])
    candidates = torch.tensor([[[1, 2, 3], [1, 2, 3]], [[2, 3, 4], [1, 2, 3]]], dtype=torch.long)
    labels = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
    loss = model(history, candidates, labels, discrete_features, real_value_features)
    print(loss)
