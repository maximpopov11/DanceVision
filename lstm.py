import lightning
import torch

SEED = 0
# 33 landmark x, y, z for 3 targets (2 targets and 0 or 0's and 1 target)
INPUT_SIZE = 33 * 3 * 3
HIDDEN_SIZE = 5
LEARNING_RATE = 0.1


class LightningLSTM(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        lightning.seed_everything(seed=SEED)
        self.lstm = torch.nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, input):
        input_trans = input.view(len(input), INPUT_SIZE)
        lstm_out, _ = self.lstm(input_trans)
        prediction = lstm_out[-1]
        return prediction

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        # TODO: use cross entropy loss weight parameter for unbalanced input classes

        # For cross entropy loss
        label_i = label_i.flatten().type(torch.LongTensor)
        output_i = output_i.unsqueeze(0).type(torch.FloatTensor)

        loss = self.loss(output_i, label_i)

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
