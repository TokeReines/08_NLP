import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from utils.optimizer import InverseSquareRootLR, LinearLR 
from utils.metric import AccMetric, F1Metric, F1Metric2

class Trainer():
    def __init__(self, fname, model, tokenizer, vocab, device, update_steps=4, clip=5, clip_all=True):
        self.device = device
        self.model = model.to(self.device)
        self.fname = fname
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.update_steps = update_steps
        self.clip = clip
        self.clip_all = clip_all

    def train(self, optimizer, scheduler, epoch, train_dataloader, dev_dataloader, test_dataloader, **kwargs):
        scheduler['steps'] = len(train_dataloader) * epoch // self.update_steps
        self.optimizer = self.init_optimizer(optimizer)
        self.scheduler = self.init_scheduler(scheduler)
        self.epoch = epoch
        self.n_batches = len(train_dataloader)
        
        best_score = 0.0
        epoch_losses = []
        for e in tqdm(range(1, self.epoch+1)):
            self.step = 1
            epoch_loss = self.train_step(train_dataloader)
            epoch_losses.append(epoch_loss)
            loss, acc = self.evaluate(train_dataloader)
            dev_loss, dev_acc = self.evaluate(dev_dataloader)
            print(f'epoch {e}: train loss = {epoch_loss}, train score: {acc}, dev loss: {dev_loss}, dev score: {dev_acc}')
            if dev_acc > best_score:
                best_score = dev_acc
                self.save_model()
                print('saved')
        return epoch_losses

    def init_optimizer(self, optimizer):
        if optimizer['name'] == 'Adam':
            optimizer = Adam(params=self.model.parameters(),
                             lr=optimizer['lr'],
                             betas=(optimizer.get('mu', 0.9), optimizer.get('nu', 0.999)),
                             eps=optimizer.get('eps', 1e-8),
                             weight_decay=optimizer.get('weight_decay', 0))
        else:
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': optimizer.get('weight_decay', 0)},
                {'params': [p for n, p in self.model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]           
            optimizer_grouped_parameters = [{'params': p, 'lr': optimizer['lr'] * (1 if n.startswith('embed') else optimizer['lr_rate']), 'weight_decay': optimizer.get('weight_decay', 0) if not any(nd in n for nd in no_decay) else 0.0} for n, p in self.model.named_parameters()]
            optimizer = AdamW(optimizer_grouped_parameters, lr=optimizer['lr'])
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
            loss, y = self.model(ques, doc, answerable)
            epoch_loss += loss.item()
            loss /= self.update_steps
            loss.backward()
            if self.step % self.update_steps == 0 or self.step % self.n_batches == 0:
                if self.clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, norm_type=2)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            self.step += 1
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
    

class SeqLabelingAnsTrainer(Trainer):

    def train(self, optimizer, scheduler, epoch, train_dataloader, dev_dataloader, test_dataloader, **kwargs):
        scheduler['steps'] = len(train_dataloader) * epoch // self.update_steps
        self.optimizer = self.init_optimizer(optimizer)
        self.scheduler = self.init_scheduler(scheduler)
        self.epoch = epoch
        self.n_batches = len(train_dataloader)
        
        best_score = 0.0
        epoch_losses = []
        for e in tqdm(range(1, self.epoch+1)):
            self.step = 1
            epoch_loss = self.train_step(train_dataloader)
            epoch_losses.append(epoch_loss)
            dev_loss, dev_acc, dev_f1 = self.evaluate(dev_dataloader)

            print(f'epoch {e}: train loss = {epoch_loss}, dev loss: {dev_loss}, dev acc: {dev_acc}, dev f1: {dev_f1}')
            if dev_f1[0] > best_score:
                best_score = dev_f1[0]
                self.save_model()
                print('saved')
        return epoch_losses

    def train_step(self, dataloader, **kwargs):
        epoch_loss = 0.0
        self.model.train()
        for i, batch in tqdm(enumerate(dataloader)):
            ques, _, ans, answerable, doc = batch
            ques, _ = ques
            doc, _ = doc
            ans, start, end = ans
            start = start.to(self.device)
            end = end.to(self.device)
            ques = ques.to(self.device)
            answerable = answerable.to(self.device)
            doc = doc.to(self.device)
        
            loss, _, _, _, _ = self.model(ques, doc, answerable, start=start, end=end)
            epoch_loss += loss.item()
            loss /= self.update_steps
            loss.backward()
            if self.step % self.update_steps == 0 or self.step % self.n_batches == 0:
                if self.clip_all:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, norm_type=2)
                else:
                    nn.utils.clip_grad_norm_([p for n, p in self.model.named_parameters() if n.startswith('classification')], self.clip, norm_type=2)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            self.step += 1
        epoch_loss /= len(dataloader)
        return epoch_loss
    
    @torch.no_grad()
    def infer(self, dataloader, **kwargs):
        self.model.eval()
        for batch in dataloader:
            # print(batch)
            ques, _, ans, answerable, doc = batch
            ques, _ = ques
            doc, _ = doc
            ans, start, end = ans
            start = start.to(self.device)
            end = end.to(self.device)
            ques = ques.to(self.device)
            answerable = answerable.to(self.device)
            doc = doc.to(self.device)
            ans = ans.to(self.device)

            loss, y_label, y_answerable, att_x, att_y = self.model(ques, doc, answerable, start=start, end=end)

            return y_label, y_answerable, att_x, att_y
        
    @torch.no_grad()
    def evaluate(self, dataloader, **kwargs):
        total_loss = 0.0
        self.model.eval()
        metric = AccMetric()
        fmetric = F1Metric()
        for batch in dataloader:
            ques, _, ans, answerable, doc = batch
            ques, _ = ques
            doc, _ = doc
            ans, start, end = ans
            start = start.to(self.device)
            end = end.to(self.device)
            ques = ques.to(self.device)
            answerable = answerable.to(self.device)
            doc = doc.to(self.device)
            ans = ans.to(self.device)

            loss, y_label, y_answerable, _, _ = self.model(ques, doc, answerable, start=start, end=end)

            y_label = self.model.decode_qa(y_label[0], y_label[1])
            if y_answerable is not None:
                y_answerable = self.model.decode(y_answerable)
                metric.update(y_answerable, answerable, None)
            fmetric.update(y_label, ans, y_answerable)
            total_loss += loss 
        
        total_loss /= len(dataloader)
        return total_loss, metric.score, fmetric.score
    