import torch.nn as nn
import torch

class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=128):
        super(FFN, self).__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        
        self.hidden_layer = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input) -> torch.Tensor:
        embedded = self.embedding(input)
        
        x = self.hidden_layer(embedded)
        x = self.relu(x)
        
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x
    
    def decode(self, x):
        y = torch.argmax(x, dim=1)
        return y