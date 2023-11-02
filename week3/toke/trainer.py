import torch
from tqdm import tqdm
import torch.nn as nn

class Trainer():
    def __init__(self, fname, model, device, tokenizer):
        self.device = device
        self.model = model.to(self.device)
        self.fname = fname
        self.tokenizer = tokenizer

    def train(self, optimizer, epoch, train_dataloader, dev_dataloader):
        self.optimizer = optimizer
        self.epoch = epoch

        best_dev_acc = 0.0
        for e in tqdm(range(1, self.epoch + 1)):
            epoch_loss = self.train_step(train_dataloader)
            dev_loss, dev_acc = self.evaluate(dev_dataloader)
            _, train_acc = self.evaluate(dev_dataloader)
            print(f'epoch {e}: train loss = {epoch_loss:.4f}, train acc: {train_acc:.2%}, dev loss: {dev_loss:.4f}, dev acc: {dev_acc:.2%}')
            
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                self.save_model()
                
    def train_step(self, dataloader):
        epoch_loss = 0.0
        self.model.train()
        for batch in tqdm(dataloader):
            input, answerable = batch
            
            answerable = answerable.to(self.device)
            input = torch.tensor(input).to(self.device)
            
            self.optimizer.zero_grad()
            
            # separator_index = self.tokenizer.vocab_dict[self.tokenizer.separator]
            # Hack to make same dimensions
            # separator = torch.tensor([separator_index]*ques.size()[0]).unsqueeze(1).to(self.device)
            # x = torch.cat([ques, doc], dim=1)
            x = input.to(self.device).float()
            
            y = self.model(x)
            
            loss = nn.functional.binary_cross_entropy(y, answerable.view(-1, 1).float())
            loss.backward()
            
            self.optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= len(dataloader)
        return epoch_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        self.model.eval()

        for batch in dataloader:
            input, answerable = batch
            
            answerable = answerable.to(self.device)
            input = torch.tensor(input).to(self.device)
            
            #separator_index = self.tokenizer.vocab_dict[self.tokenizer.separator]
            # Hack to make same dimensions
            #separator = torch.tensor([separator_index]*ques.size()[0]).unsqueeze(1).to(self.device)
            
            # x = torch.cat([ques, doc], dim=1)
            x = input.to(self.device).float()
            y = self.model(x)
            
            loss = nn.functional.binary_cross_entropy(y, answerable.view(-1, 1).float())
            total_loss += loss.item()

            predicted_labels = (y > 0.5).int()
            correct_predictions += (predicted_labels.squeeze(-1) == answerable).sum().item()
            total_samples += answerable.size(0)

        accuracy = correct_predictions / total_samples
        total_loss /= len(dataloader)
        return total_loss, accuracy

    def save_model(self):
        torch.save(self.model.state_dict(), self.fname)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    