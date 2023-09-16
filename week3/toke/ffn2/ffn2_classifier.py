import torch.nn as nn
import torch
from transformers import BertModel

class FFN2Classifier(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', hidden_dim=768, device='cuda'):
        super(FFN2Classifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.fc1 = nn.Linear(2 * self.bert.config.hidden_size, hidden_dim)  # x2 since we concatenate question and document embeddings
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, question_ids, question_mask, document_ids, document_mask):
        question_emb = self.bert(question_ids, attention_mask=question_mask).last_hidden_state.mean(dim=1)
        document_emb = self.bert(document_ids, attention_mask=document_mask).last_hidden_state.mean(dim=1)
        
        x = torch.cat((question_emb, document_emb), dim=-1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return self.sigmoid(x)