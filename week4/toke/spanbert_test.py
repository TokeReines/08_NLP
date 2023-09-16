import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForQuestionAnswering

# Pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Document and question
document = "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."
question = "Who developed the theory of relativity?"

# Tokenize the input
tokens = tokenizer(question, document, return_tensors="pt", padding=True, truncation=True)

# Forward pass through the model
outputs = model(**tokens)

start_logits, end_logits = outputs.start_logits, outputs.end_logits

# Convert logits to probabilities
start_probs = torch.softmax(start_logits, dim=1)[0]
end_probs = torch.softmax(end_logits, dim=1)[0]

# Find the start and end positions with the highest probability
start_idx = torch.argmax(start_probs)
end_idx = torch.argmax(end_probs)

# Convert token positions to text span
answer_span = tokens.input_ids[0][start_idx:end_idx+1]
answer_text = tokenizer.decode(answer_span)

print("Predicted Answer:", answer_text)