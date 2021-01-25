import torch
import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, vocab_size, emb_dims):
        super().__init__()
        self.embedding_in = nn.Embedding(vocab_size, emb_dims)
        self.embedding_out = nn.Embedding(vocab_size, emb_dims)
        init_range = 0.5 / emb_dims
        self.embedding_in.weight.data.uniform_(-init_range, init_range)
        self.embedding_out.weight.data.uniform_(-init_range, init_range)

    def forward(self, central_items, context_items):
        hidden = self.embedding_in(central_items)
        context = self.embedding_out(context_items)
        logits = torch.bmm(hidden.unsqueeze(1), context.permute(0, 2, 1))
        return torch.sigmoid(logits.squeeze(1))


class GESModel(nn.Module):
    def __init__(self, num_items, side_info_vocab_sizes, emb_dims=100):
        super().__init__()
        self.emb_dims = emb_dims
        self.item_embedding_in = nn.Embedding(num_items, emb_dims)
        self.item_embedding_out = nn.Embedding(num_items, emb_dims)
        embeddings = []
        for vocab_size in side_info_vocab_sizes:
            embeddings.append(nn.Embedding(vocab_size, emb_dims))
        self.side_info_embeddings = nn.ModuleList(embeddings)
        self._init_embedding()

    def _init_embedding(self):
        init_range = 0.5 / self.emb_dims
        self.item_embedding_in.weight.data.uniform_(-init_range, init_range)
        self.item_embedding_out.weight.data.uniform_(-init_range, init_range)
        for embedding in self.side_info_embeddings:
            embedding.weight.data.uniform_(-init_range, init_range)

    def forward(self, central_items, central_side_informations, context_items):
        hidden = self._emb_items(central_items, central_side_informations)
        context = self.item_embedding_out(context_items)
        logits = torch.bmm(hidden.unsqueeze(1), context.permute(0, 2, 1))
        return torch.sigmoid(logits.squeeze(1))

    def _emb_items(self, items, side_informations):
        item_embs = self.item_embedding_in(items)
        side_info_embs = [self.side_info_embeddings[i](side_info) for i, side_info in enumerate(side_informations)]
        embs = item_embs
        for i in range(len(side_info_embs)):
            if len(side_info_embs[i].shape) == 3:
                side_info_embs[i] = side_info_embs[i].mean(dim=1)
            embs = embs + side_info_embs[i]
        embs = embs / (len(side_info_embs) + 1)
        return embs


class EGESModel(nn.Module):
    def __init__(self, num_items, side_info_vocab_sizes, emb_dims=100):
        super().__init__()
        self.emb_dims = emb_dims
        self.item_embedding_in = nn.Embedding(num_items, emb_dims)
        self.item_embedding_out = nn.Embedding(num_items, emb_dims)
        self.weights = nn.Embedding(num_items, len(side_info_vocab_sizes)+1)
        embeddings = []
        for vocab_size in side_info_vocab_sizes:
            embeddings.append(nn.Embedding(vocab_size, emb_dims))
        self.side_info_embeddings = nn.ModuleList(embeddings)
        self._init_embedding()

    def _init_embedding(self):
        init_range = 0.5 / self.emb_dims
        self.item_embedding_in.weight.data.uniform_(-init_range, init_range)
        self.item_embedding_out.weight.data.uniform_(-init_range, init_range)
        for embedding in self.side_info_embeddings:
            embedding.weight.data.uniform_(-init_range, init_range)

    def forward(self, central_items, central_side_informations, context_items):
        hidden = self._emb_items(central_items, central_side_informations)
        context = self.item_embedding_out(context_items)
        logits = torch.bmm(hidden.unsqueeze(1), context.permute(0, 2, 1))
        return torch.sigmoid(logits.squeeze(1))

    def _emb_items(self, items, side_informations):
        item_embs = self.item_embedding_in(items)
        side_info_embs = [self.side_info_embeddings[i](side_info) for i, side_info in enumerate(side_informations)]
        for i in range(len(side_info_embs)):
            if len(side_info_embs[i].shape) == 3:
                side_info_embs[i] = side_info_embs[i].mean(dim=1)
        batch_size = items.shape[0]
        embs = torch.cat([item_embs] + side_info_embs, dim=-1)
        embs = embs.reshape(batch_size, -1, self.emb_dims)
        item_weights = self.weights(items)
        item_weights = torch.softmax(item_weights, dim=1)
        embs = torch.bmm(item_weights.unsqueeze(1), embs)
        embs = embs.squeeze(1)
        return embs    


if __name__ == "__main__":
    # model = SkipGram(5, 10)
    # items = torch.tensor([2, 1], dtype=torch.long)
    # context = torch.tensor([[1, 3, 4], [2, 3, 4]], dtype=torch.long)
    # print(model(items, context))

    model = EGESModel(5, [5, 4], 5)
    items = torch.tensor([2, 1], dtype=torch.long)
    side_infos = [torch.tensor([[1, 2], [2, 3]], dtype=torch.long), torch.tensor([1, 2])]

    contexts = torch.tensor([[1, 3, 2], [2, 4, 1]], dtype=torch.long)
    context_side_infos = [torch.tensor([[[1, 2], [2, 3]], [[1, 2], [2, 3]]], dtype=torch.long), torch.tensor([[1, 2], [1, 2]])]

    output = model(items, side_infos, contexts)
    print(output)
