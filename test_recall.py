from recommendation.dataset.recall import YoutubeNetDataset, GESDataset
from recommendation.model.recall import YoutubeNetModel, MhaRecallModel
import tqdm
import torch
from torch.utils.data import DataLoader
import json
import os.path
from recall import Item2vectorRecaller


def train(model, optimizer, data_loader, device, log_interval=100, model_type="youtube"):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, values in enumerate(tk0):
        values = [value.to(device) for value in values]
        if len(values) == 3:
            history, history_sizes, positives = values
            discretes = None
        else:
            history, history_sizes, positives = values[:3]
            discretes = values[3:]
        if model_type == "youtube":
            loss = model(history, positives, discrete_features=discretes)
        else:
            loss = model(history, history_sizes, positives, discrete_features=discretes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def train_model(model_type):
    dataset = RecallDataset("data/positive_ratings.txt")

    save_dir = "data/model"
    with open(os.path.join(save_dir, "vocab.txt"), "w", encoding="utf-8") as fo:
        json.dump(dataset.vocab, fo)
    with open(os.path.join(save_dir, "attr_vocabs.txt"), "w", encoding="utf-8") as fo:
        json.dump(dataset.attr_vocabs, fo)
    num_items = len(dataset.vocab) + 2
    emb_dims = 100
    if model_type == "youtube":
        model = YoutubeNetModel(num_items, emb_dims, weights=dataset.weights)
    else:
        model = MhaRecallModel(num_items, emb_dims, 10, weights=dataset.weights)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=1e-6)
    device = "cpu"
    batch_size = 32
    train_data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    for _ in range(3):
        train(model, optimizer, train_data_loader, device, model_type="mha")
    if model_type == "youtube":
        model_path = os.path.join(save_dir, "youtube_net.pt")
    else:
        model_path = os.path.join(save_dir, "mha.pt")
    torch.save(model, model_path)
    with open(os.path.join(save_dir, "item_vecs.txt"), "w", encoding="utf-8") as fo:
        fo.write(f"{model.num_items} {model.item_emb_dims}\n")
        weights = model.loss.embdding.weight.cpu().detach().numpy()
        for content_id, idx in dataset.vocab.items():
            vec = ['0.7%f' % weight for weight in weights[idx]]
            fo.write(f"{content_id} {' '.join(vec)}\n")


def test_model(model_dir, model_type, test_file, min_history, max_history, device="cpu", topk=20):
    recaller = Item2vectorRecaller(model_dir)
    recaller.build_index([])
    if model_type == "youtube":
        model_path = os.path.join(model_dir, "youtube_net.pt")
    else:
        model_path = os.path.join(model_dir, "mha.pt")
    with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as fi:
        vocab = json.load(fi)
    model = torch.load(model_path, map_location=device)
    with open(test_file) as fi:
        total = 0
        hit = 0
        n = 0
        for line in fi:
            _, clicked = line.strip().split("\t")
            clicked_ids = clicked.split(" ")
            history_list = []
            history_sizes = []
            targets = []
            for i in range(min_history, len(clicked_ids)):
                history = clicked_ids[max(0, i-max_history):i]
                history = [vocab.get(content_id, len(vocab)+1) for content_id in history]
                history_sizes.append(len(history))
                history += [0] * (max_history - len(history))
                history_list.append(history)
                targets.append(clicked_ids[i])
            history_tensor = torch.tensor(history_list, dtype=torch.long, device=device)
            if model_type == "youtube":
                vecs = model(history_tensor)
            else:
                history_sizes = torch.tensor(history_sizes, device=device)
                vecs = model(history_tensor, history_sizes)
            vecs = vecs.cpu().detach().numpy()
            recalled = recaller.index.retrieve(vecs, topk)
            n += 1
            total += len(targets)
            for i, target in enumerate(targets):
                if target in recalled[i]:
                    hit += 1
            if n == 1:
                break
        print(hit / total)


if __name__ == "__main__":
    train_model()
