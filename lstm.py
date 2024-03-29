import torch.nn as nn
from torch.optim import Adam

import lightning

SEED = 0
INPUT_SIZE = 33 * 2
HIDDEN_SIZE = 5
LEARNING_RATE = 0.1


class LightningLSTM(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        lightning.seed_everything(seed=SEED)
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)

    def forward(self, input):
        for frame in input:
            input_trans = frame.view(len(frame), 1)
            lstm_out, temp = self.lstm(input_trans)
        prediction = lstm_out[-1]
        return prediction

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=LEARNING_RATE)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss)

        if label_i == 0:
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss
