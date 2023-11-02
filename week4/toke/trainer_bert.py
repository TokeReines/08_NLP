import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score


class TrainerBERT():
    def __init__(self, fname, model, device, optimizer, scheduler, tokenizer):
        self.device = device
        self.model = model.to(self.device)
        self.fname = fname
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.tokenizer = tokenizer

    def train(self, epoch, train_dataloader, dev_dataloader):
        self.epoch = epoch

        best_dev_acc = 0.0
        for e in tqdm(range(1, self.epoch + 1)):
            epoch_loss, (train_em, train_f1) = self.train_step(
                train_dataloader)
            dev_em, dev_f1 = self.evaluate(dev_dataloader)

            print(
                f'epoch {e}: train loss = {epoch_loss:.4f}, train EM: {train_em:.2%}, train f1: {train_f1}')
            print(f'epoch {e}: dev EM: {dev_em:.2%}, dev f1: {dev_f1}')

            if dev_f1 > best_dev_acc:
                best_dev_acc = dev_f1
                self.save_model()

    def train_step(self, dataloader):
        epoch_loss = 0.0
        self.model.train()

        total_em, total_f1, total_count = 0, 0, len(dataloader)

        for batch in tqdm(dataloader):
            input, labels = batch
            # Transfer to device
            for k, v in input.items():
                input[k] = v.to(self.device)

            for k, v in labels.items():
                labels[k] = v.to(self.device)

            self.optimizer.zero_grad()

            # Call the model
            outputs = self.model(
                input_ids=input['input_ids'][:, 0],
                token_type_ids=input['token_type_ids'][:, 0],
                attention_mask=input['attention_mask'][:, 0],
                start_positions=labels['answer_start'],
                end_positions=labels['answer_end']
            )
            # Extract outputs
            loss = outputs.loss
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            loss.backward()
            epoch_loss += loss.item()

            self.optimizer.step()
            self.scheduler.step()

            pred_start, pred_end = start_logits.argmax(
                dim=1), end_logits.argmax(dim=1)
            true_start, true_end = labels['answer_start'], labels['answer_end']

            # For EM
            batch_em = sum(self.compute_exact_match(ps, pe, ts, te) for ps, pe, ts, te in zip(
                pred_start, pred_end, true_start, true_end))
            total_em += batch_em

            # For F1
            for ps, pe, ts, te in zip(pred_start, pred_end, true_start, true_end):
                pred_answer = self.tokenizer.tokenizer.decode(
                    input['input_ids'][0][0][ps:pe+1])
                true_answer = self.tokenizer.tokenizer.decode(
                    input['input_ids'][0][0][ts:te+1])

                pred_tokens = self.tokenizer.tokenizer.tokenize(pred_answer)
                true_tokens = self.tokenizer.tokenizer.tokenize(true_answer)

                total_f1 += self.compute_f1(pred_tokens, true_tokens)

        epoch_loss /= len(dataloader)
        return epoch_loss, (total_em / total_count, total_f1 / total_count)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()

        total_em, total_f1, total_count = 0, 0, len(dataloader)
        for batch in dataloader:
            input, labels = batch
            for k, v in input.items():
                input[k] = v.to(self.device)

            for k, v in labels.items():
                labels[k] = v.to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input['input_ids'][:, 0],
                    token_type_ids=input['token_type_ids'][:, 0],
                    attention_mask=input['attention_mask'][:, 0],
                    start_positions=labels['answer_start'],
                    end_positions=labels['answer_end']
                )

            # Extract outputs
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            pred_start, pred_end = start_logits.argmax(
                dim=1), end_logits.argmax(dim=1)
            true_start, true_end = labels['answer_start'], labels['answer_end']

            # For EM
            batch_em = sum(self.compute_exact_match(ps, pe, ts, te) for ps, pe, ts, te in zip(
                pred_start, pred_end, true_start, true_end))
            total_em += batch_em

            # For F1
            for ps, pe, ts, te in zip(pred_start, pred_end, true_start, true_end):
                pred_answer = self.tokenizer.tokenizer.decode(
                    input['input_ids'][0][0][ps:pe+1])
                true_answer = self.tokenizer.tokenizer.decode(
                    input['input_ids'][0][0][ts:te+1])

                pred_tokens = self.tokenizer.tokenizer.tokenize(pred_answer)
                true_tokens = self.tokenizer.tokenizer.tokenize(true_answer)

                total_f1 += self.compute_f1(pred_tokens, true_tokens)

        return total_em / total_count, total_f1 / total_count

    def compute_exact_match(self, pred_start, pred_end, true_start, true_end):
        return int(pred_start == true_start and pred_end == true_end)

    def compute_f1(self, pred_tokens, true_tokens):
        common = set(pred_tokens) & set(true_tokens)

        if not pred_tokens or not true_tokens:
            # If either the predicted or true answer is no-answer, then F1 is 0
            return 0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(true_tokens)

        if precision + recall == 0:
            return 0

        f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    def save_model(self):
        torch.save(self.model.state_dict(), self.fname)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
