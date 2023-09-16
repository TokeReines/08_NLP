import torch.nn as nn

# Define the feed forward network


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()

        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.BCELoss()

    def forward(self, input, label) -> tuple[float, float]:
        x = self.hidden_layer(input)
        x = self.relu(x)

        x = self.output_layer(x)
        y = self.sigmoid(x)

        loss = self.criterion(y, label)

        return loss, y
