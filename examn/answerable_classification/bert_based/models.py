import torch.nn as nn
import torch
from transformers import BertModel, AutoModelForQuestionAnswering

class MeanClassifier(nn.Module):
    def __init__(self, pretrained_model, hidden_dim=50):
        super(MeanClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2 * self.bert.config.hidden_size, hidden_dim)  # x2 since we concatenate question and document embeddings
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, question_ids, question_mask, document_ids, document_mask):
        question_emb = self.bert(question_ids, attention_mask=question_mask).last_hidden_state.mean(dim=1)
        document_emb = self.bert(document_ids, attention_mask=document_mask).last_hidden_state.mean(dim=1)

        x = torch.cat((question_emb, document_emb), dim=-1)        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)


class CLSClassifier(nn.Module):
    def __init__(self, pretrained_model, hidden_dim=50):
        super(CLSClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2 * self.bert.config.hidden_size, hidden_dim)  # Use only the BERT hidden size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, question_ids, question_mask, document_ids, document_mask):
        # Extract the embeddings for the [CLS] token (first token)
        question_cls_emb = self.bert(question_ids, attention_mask=question_mask).last_hidden_state[:, 0, :]
        document_cls_emb = self.bert(document_ids, attention_mask=document_mask).last_hidden_state[:, 0, :]
        
        x = torch.cat((question_cls_emb, document_cls_emb), dim=-1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        y = self.sigmoid(x)
        return y
    
class LogRegCLSClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(LogRegCLSClassifier, self).__init__()
        self.bert = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.logreg = nn.Linear(2 * self.bert.config.hidden_size, 1)  # Use only the BERT hidden size
        self.sigmoid = nn.Sigmoid()

    def forward(self, question_ids, question_mask, document_ids, document_mask, output_attentions=False):
        # Extract the embeddings for the [CLS] token (first token)
        question_output = self.bert(question_ids, attention_mask=question_mask, output_attentions=output_attentions)
        question_cls_emb = question_output.last_hidden_state[:, 0, :]
        
        document_output = self.bert(document_ids, attention_mask=document_mask, output_attentions=output_attentions)
        document_cls_emb = document_output.last_hidden_state[:, 0, :]

        
        x = torch.cat((question_cls_emb, document_cls_emb), dim=-1)
        x = self.logreg(x)
        y = self.sigmoid(x)
        return y