import torch.nn as nn
from torch.optim import Adam

import lightning

SEED = 0
# 33 landmark x, y, z for 3 targets (2 targets and 0 or 0's and 1 target)
INPUT_SIZE = 33 * 3 * 3
HIDDEN_SIZE = 5
LEARNING_RATE = 0.1


class LightningLSTM(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        lightning.seed_everything(seed=SEED)
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)

    def forward(self, input):
        input_trans = input.view(len(input), INPUT_SIZE)
        lstm_out, _ = self.lstm(input_trans)
        prediction = lstm_out[-1]
        return prediction

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=LEARNING_RATE)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        # TODO: What is the best loss function?
        # TODO: Should the prediction be the greatest output? Output closest to 1? Other?
        loss = 0
        for i in range(5):
            if i == label_i:
                loss += (1 - output_i[i]) ** 2
            # else:
            #     loss += (output_i[i]) ** 2

        # loss = (output_i - label_i) ** 2

        # TODO: graph training results
        # self.log("train_loss", loss)
        #
        # if label_i == 0:
        #     self.log("out_0", output_i)
        # elif label_i == 1:
        #     self.log("out_1", output_i)
        # elif label_i == 2:
        #     self.log("out_2", output_i)
        # elif label_i == 3:
        #     self.log("out_3", output_i)
        # elif label_i == 4:
        #     self.log("out_4", output_i)
        # else:
        #     print("label out of scope: ", label_i)

        return loss
