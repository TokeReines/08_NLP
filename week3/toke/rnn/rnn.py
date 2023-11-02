import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        self.criterion = nn.BCELoss()

    def forward(self, input_sequence, label):
        # embedded = self.embedding(input_sequence.long())
        output, _ = self.rnn(input_sequence)
        # output = output[-1]  # Take the last output only
        output = self.fc(output)
        y = self.sigmoid(output)
        
        loss = self.criterion(y, label)

        return loss, y
    