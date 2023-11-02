import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        self.criterion = nn.BCELoss()

    def forward(self, input, label):        
        x = self.linear(input)
        y = self.sigmoid(x)
        
        loss = self.criterion(y, label)

        return loss, y