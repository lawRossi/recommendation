from recommendation.dataset.recall import RecallDataset
from recommendation.model.recall import YoutubeNetModel, MhaRecallModel
import tqdm
import torch
from torch.utils.data import DataLoader


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


if __name__ == "__main__":
    dataset = RecallDataset("data/positive_ratings.txt")
    num_items = len(dataset.vocab) + 2
    emb_dims = 100
    # model = YoutubeNetModel(num_items, emb_dims)
    model = MhaRecallModel(num_items, emb_dims, 10)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=1e-6)
    device = "cpu"
    batch_size = 32
    train_data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    for epoch_i in range(3):
        train(model, optimizer, train_data_loader, device, model_type="mha")
