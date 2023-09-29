import torch
from tqdm import tqdm
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from utils.optimizer import InverseSquareRootLR, LinearLR 
from utils.metric import AccMetric

class Trainer():
    def __init__(self, fname, model, tokenizer, vocab, device):
        self.device = device
        self.model = model.to(self.device)
        self.fname = fname
        self.tokenizer = tokenizer
        self.vocab = vocab

    def train(self, optimizer, scheduler, epoch, train_dataloader, dev_dataloader, test_dataloader, **kwargs):
        scheduler['steps'] = len(train_dataloader) * epoch
        self.optimizer = self.init_optimizer(optimizer)
        self.scheduler = self.init_scheduler(scheduler)
        self.epoch = epoch
        
        best_score = 0.0
        epoch_losses = []
        for e in tqdm(range(1, self.epoch+1)):
            epoch_loss = self.train_step(train_dataloader)
            epoch_losses.append(epoch_loss)
            loss, acc = self.evaluate(train_dataloader)
            dev_loss, dev_acc = self.evaluate(dev_dataloader)
            print(f'epoch {e}: train loss = {epoch_loss}, train acc: {acc}, dev loss: {dev_loss}, dev acc: {dev_acc}')
            if dev_acc > best_score:
                best_score = dev_acc
                self.save_model()
                print('saved')
        return epoch_losses
        # loss, acc = self.evaluate(test_dataloader)
        # print(f'test loss = {loss},test acc: {acc}')

    def init_optimizer(self, optimizer):
        if optimizer['name'] == 'Adam':
            optimizer = Adam(params=self.model.parameters(),
                             lr=optimizer['lr'],
                             betas=(optimizer.get('mu', 0.9), optimizer.get('nu', 0.999)),
                             eps=optimizer.get('eps', 1e-8),
                             weight_decay=optimizer.get('weight_decay', 0))
        else:
            optimizer = AdamW(params=[{'params': p, 'lr': optimizer['lr'] * (1 if n.endswith('embed') else optimizer['lr_rate'])}
                                      for n, p in self.model.named_parameters()],
                              lr=optimizer['lr'],
                              betas=(optimizer.get('mu', 0.9), optimizer.get('nu', 0.999)),
                              eps=optimizer.get('eps', 1e-8),
                              weight_decay=optimizer.get('weight_decay', 0))
        return optimizer
    
    def init_scheduler(self, scheduler):
        if scheduler['name'] == 'linear':
            scheduler = LinearLR(optimizer=self.optimizer,
                                 warmup_steps=scheduler.get('warmup_steps', int(scheduler['steps']*scheduler.get('warmup', 0))),
                                 steps=scheduler['steps'])
        elif scheduler['name'] == 'inverse':
            scheduler = InverseSquareRootLR(optimizer=self.optimizer,
                                            warmup_steps=scheduler['warmup_steps'])
        else:
            scheduler = ExponentialLR(optimizer=self.optimizer,
                                      gamma=scheduler['decay']**(1/scheduler['decay_steps']))
        return scheduler
    
    def train_step(self, dataloader, **kwargs):
        epoch_loss = 0.0
        self.model.train()
        for i, batch in tqdm(enumerate(dataloader)):
            ques, ans_start, ans, answerable, doc = batch
            ques = ques.to(self.device)
            answerable = answerable.to(self.device)
            doc = doc.to(self.device)
            self.optimizer.zero_grad()
            loss, y = self.model(ques, doc, answerable)
            epoch_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        epoch_loss /= len(dataloader)
        return epoch_loss

    @torch.no_grad()
    def evaluate(self, dataloader, **kwargs):
        total_loss = 0.0
        self.model.eval()
        metric = AccMetric()
        for batch in dataloader:
            ques, ans_start, ans, answerable, doc = batch
            ques = ques.to(self.device)
            answerable = answerable.to(self.device)
            doc = doc.to(self.device)
            loss, y = self.model(ques, doc, answerable)
            y_pred = self.model.decode(y)
            total_loss += loss 
            metric.update(y_pred, answerable, None)
        
        total_loss /= len(dataloader)
        return total_loss, metric.score

    def save_model(self):
        torch.save(self.model.state_dict(), self.fname)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    


