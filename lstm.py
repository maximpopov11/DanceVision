import lightning
import torch

from torch.utils.tensorboard import SummaryWriter

SEED = 0
# 33 landmark x, y, z for 3 targets (2 targets and 0 or 0's and 1 target)
INPUT_SIZE = 33 * 3 * 3
HIDDEN_SIZE = 5
LEARNING_RATE = 0.01


class LightningLSTM(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        lightning.seed_everything(seed=SEED)
        self.lstm = torch.nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
        self.loss = torch.nn.CrossEntropyLoss()
        self.writer = SummaryWriter()

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

        # For cross entropy loss
        label_i = label_i.flatten().type(torch.LongTensor)
        output_i = output_i.unsqueeze(0).type(torch.FloatTensor)

        loss = self.loss(output_i, label_i)

        if label_i == 0:
            self.writer.add_scalar("0", loss, self.current_epoch)
        elif label_i == 1:
            self.writer.add_scalar("1", loss, self.current_epoch)
        elif label_i == 2:
            self.writer.add_scalar("2", loss, self.current_epoch)
        elif label_i == 3:
            self.writer.add_scalar("3", loss, self.current_epoch)
        elif label_i == 4:
            self.writer.add_scalar("4", loss, self.current_epoch)
        else:
            print("label out of scope: ", label_i)
        self.writer.flush()

        return loss
