import lightning
import torch

from data_parsing import labels
from feature_generation import generate_features
from lstm import LightningLSTM
from torch.utils.data import TensorDataset, DataLoader

VIDEO_PATHS = ["v0_" + str(i) + ".mp4" for i in range(len(labels[0]))]
EPOCHS = 300
LOG_STEPS = 1


if __name__ == '__main__':
    # TODO: train/test sets
    features = torch.tensor(generate_features(VIDEO_PATHS))
    labels = torch.tensor(labels)

    dataset = TensorDataset(features, labels[0])
    dataloader = DataLoader(dataset)

    model = LightningLSTM()
    trainer = lightning.Trainer(max_epochs=EPOCHS, log_every_n_steps=LOG_STEPS)
    trainer.fit(model, train_dataloaders=dataloader)

    # TODO: print results nicely
    for i in range(len(labels)):
        print("Sample " + str(i) + ", Label: " + labels[0][i].clone().detach() + ", Predicted: ", model(features[0][i].clone().detach()))
