import lightning
import torch

from lstm import LightningLSTM
from torch.utils.data import TensorDataset, DataLoader


if __name__ == '__main__':
    dummy_features = torch.tensor([
        [
            [1.]*33 + [2.] * 33,
            [2.]*33 + [1.] * 33,
            [1.]*33 + [2.] * 33
        ],
        [
            [1.]*33 + [2.] * 33,
            [1.]*33 + [1.] * 33,
            [1.]*33 + [2.] * 33
        ],
        [
            [1.]*33 + [2.] * 33,
            [-1.]*33 + [1.] * 33,
            [1.]*33 + [0.] * 33
        ],
        [
            [1.]*33 + [2.] * 33,
            [1.]*33 + [1.] * 33,
            [1.]*33 + [0.] * 33
        ],
        [
            [1.]*33 + [2.] * 33,
            [-1.]*33 + [1.] * 33,
            [1.]*33 + [2.] * 33
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
    print("Sample 0: Observed = 0, Predicted =", model(dummy_features[0].clone().detach()))
    print("Sample 1: Observed = 1, Predicted =", model(dummy_features[1].clone().detach()))
    print("Sample 2: Observed = 2, Predicted =", model(dummy_features[2].clone().detach()))
    print("Sample 3: Observed = 3, Predicted =", model(dummy_features[3].clone().detach()))
    print("Sample 4: Observed = 4, Predicted =", model(dummy_features[4].clone().detach()))

    trainer = lightning.Trainer(max_epochs=300, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=dataloader)

    print("After optimization, the parameters are...")
    for name, param in model.named_parameters():
        print(name, param.data)

    print("\nNow let's compare the observed and predicted values after optimization...")
    print("Sample 0: Observed = 0, Predicted =", model(dummy_features[0].clone().detach()))
    print("Sample 1: Observed = 1, Predicted =", model(dummy_features[1].clone().detach()))
    print("Sample 2: Observed = 2, Predicted =", model(dummy_features[2].clone().detach()))
    print("Sample 3: Observed = 3, Predicted =", model(dummy_features[3].clone().detach()))
    print("Sample 4: Observed = 4, Predicted =", model(dummy_features[4].clone().detach()))
