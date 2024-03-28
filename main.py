import lightning
import os
import torch

from lstm import LightningLSTM
from pytube import YouTube
from torch.utils.data import TensorDataset, DataLoader


def download_video():
    YouTube('https://www.youtube.com/watch?v=G5ervgot15Y').streams.first().download(os.getcwd())


if __name__ == '__main__':
    dummy_features = torch.tensor([
        [
            [[1]*33 + [2] * 33],
            [[2]*33 + [1] * 33],
            [[1]*33 + [2] * 33]
        ],
        [
            [[1]*33 + [2] * 33],
            [[1]*33 + [1] * 33],
            [[1]*33 + [2] * 33]
        ],
        [
            [[1]*33 + [2] * 33],
            [[-1]*33 + [1] * 33],
            [[1]*33 + [0] * 33]
        ],
        [
            [[1]*33 + [2] * 33],
            [[1]*33 + [1] * 33],
            [[1]*33 + [0] * 33]
        ],
        [
            [[1]*33 + [2] * 33],
            [[-1]*33 + [1] * 33],
            [[1]*33 + [2] * 33]
        ],
    ])
    # 0 = other
    # 1 = sugar
    # 2 = left
    # 3 = right
    # 4 = whip
    dummy_labels = torch.tensor([0, 1, 2, 3, 4])

    dataset = TensorDataset(dummy_features, dummy_labels)
    dataloader = DataLoader(dataset)

    model = LightningLSTM()

    print("Before optimization, the parameters are...")
    for name, param in model.named_parameters():
        print(name, param.data)

    print("\nNow let's compare the observed and predicted values before optimization...")
    print("Sample 0 (other): Observed = 0, Predicted =", model(torch.tensor(dummy_features[0])).detach())
    print("Sample 1 (sugar): Observed = 1, Predicted =", model(torch.tensor(dummy_features[1])).detach())
    print("Sample 2 (left): Observed = 2, Predicted =", model(torch.tensor(dummy_features[2])).detach())
    print("Sample 3 (right): Observed = 3, Predicted =", model(torch.tensor(dummy_features[3])).detach())
    print("Sample 4 (whip): Observed = 4, Predicted =", model(torch.tensor(dummy_features[4])).detach())

    trainer = lightning.Trainer(max_epochs=300, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=dataloader)

    print("After optimization, the parameters are...")
    for name, param in model.named_parameters():
        print(name, param.data)

    print("\nNow let's compare the observed and predicted values after optimization...")
    print("Sample 0 (other): Observed = 0, Predicted =", model(torch.tensor(dummy_features[0])).detach())
    print("Sample 1 (sugar): Observed = 1, Predicted =", model(torch.tensor(dummy_features[1])).detach())
    print("Sample 2 (left): Observed = 2, Predicted =", model(torch.tensor(dummy_features[2])).detach())
    print("Sample 3 (right): Observed = 3, Predicted =", model(torch.tensor(dummy_features[3])).detach())
    print("Sample 4 (whip): Observed = 4, Predicted =", model(torch.tensor(dummy_features[4])).detach())
