import torch.nn as nn
import torch

class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFN, self).__init__()
        
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, questions, documents) -> torch.Tensor:
        x = torch.cat([questions, documents], dim=1)
        # x = documents
        x = self.hidden_layer(x)
        x = self.relu(x)
        
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x
    
    def decode(self, x):
        y = torch.argmax(x, dim=1)
        return y