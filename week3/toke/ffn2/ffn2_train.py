import torch_directml
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from datasets import load_dataset
from week3.toke.ffn2.ffn2_classifier import FFN2Classifier
from week3.toke.ffn2.ffn2_dataset import FFN2Dataset

from week3.toke.ffn2.ffn2_tokenizer import FFN2Tokenizer
from tqdm import tqdm


def train(model, optimizer, dataset, validation_set, epochs=5, batch_size=32, lr=0.001, device='cuda'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    model.to(device)
    

    for epoch in range(epochs):
        epoch_loss = None
        model.train()
        for q_ids, q_mask, d_ids, d_mask, labels in tqdm(dataloader, desc=f"Training, Epoch {epoch+1}/{epochs}"):
            q_ids, q_mask, d_ids, d_mask, labels = q_ids.to(device).squeeze(1), q_mask.to(
                device).squeeze(1), d_ids.to(device).squeeze(1), d_mask.to(device).squeeze(1), labels.to(device)

            optimizer.zero_grad()
            outputs = model(q_ids, q_mask, d_ids, d_mask).squeeze()
            loss = criterion(outputs, labels.float())

            loss.backward()
            
            if epoch_loss is None:
                epoch_loss = loss.item()
            else:
                epoch_loss += loss.item()
                
            optimizer.step()
            
        epoch_loss /= len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')
        
        evaluate(model, validation_set, batch_size=batch_size, device=device)


def evaluate(model, dataset, batch_size=32, device='cuda'):
    print("Evaluating")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.BCELoss()
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for q_ids, q_mask, d_ids, d_mask, labels in tqdm(dataloader, desc=f"Evaluating"):
        q_ids, q_mask, d_ids, d_mask, labels = q_ids.to(device).squeeze(1), q_mask.to(
            device).squeeze(1), d_ids.to(device).squeeze(1), d_mask.to(device).squeeze(1), labels.to(device)

        outputs = model(q_ids, q_mask, d_ids, d_mask).squeeze()
        loss = criterion(outputs, labels.float())
        total_loss += loss.item()

        predicted_labels = (outputs > 0.5).int()
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
        
    total_loss /= len(dataloader)
    accuracy = correct_predictions / total_samples
    print(f'Loss: {total_loss}, Accuracy: {accuracy:.2%}')

language = "english"
bert = 'bert-base-uncased'
device = torch_directml.device()
input_dim = 768
hidden_dim = 50
lr = 1e-3
batch_size = 8
epochs = 5

dataset = load_dataset("copenlu/answerable_tydiqa")
dataset = dataset.filter(lambda row: row['language'] == language)

train_set = dataset["train"].select(range(500))
validation_set = dataset["validation"].select(range(500))

tokenizer = FFN2Tokenizer(bert)
train_set = FFN2Dataset(train_set, tokenizer)
validation_set = FFN2Dataset(validation_set, tokenizer)
model = FFN2Classifier(bert, device=device)
optimizer = Adam(model.parameters(), lr=lr)

train(model, optimizer, train_set, validation_set, epochs=epochs, batch_size=batch_size, lr=lr, device=device)
