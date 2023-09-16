import torch
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from collections import defaultdict, OrderedDict

def get_train_features(tk, samples):
    '''
    Tokenizes all of the text in the given samples, splittling inputs that are too long for our model
    across multiple features. Finds the token offsets of the answers, which serve as the labels for
    our inputs.
    '''
    batch = tk.batch_encode_plus(
        [[q, c] for q, c in zip(samples['question_text'],
                                samples['document_plaintext'])],
        padding='max_length',
        truncation='only_second',
        stride=50,
        max_length=300,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )
    
    # Let's handle overflowing sequences
    overflowing_labels = []
    sample_map = batch.get('overflow_to_sample_mapping', range(len(samples['annotations'])))  # use range as default for no overflowing
    
    for idx in sample_map:
        label = int(samples['annotations'][idx]['answer_start'][0] != -1)
        overflowing_labels.append(label)

    batch['labels'] = torch.tensor(overflowing_labels).long()

    assert len(overflowing_labels) == len(batch['input_ids']), f"Expected {len(batch['input_ids'])} labels but got {len(batch['labels'] )}."
    return batch


def get_validation_features(tk, samples):
    # First, tokenize the text. We get the offsets and return overflowing sequences in
    # order to break up long sequences into multiple inputs. The offsets will help us
    # determine the original answer text
    batch = tk.batch_encode_plus(
        [[q, c] for q, c in zip(samples['question_text'],
                                samples['document_plaintext'])],
        padding='max_length',
        truncation='only_second',
        stride=50,
        max_length=300,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )
    # We'll store the ID of the samples to calculate squad score
    batch['example_id'] = []
    # The overflow sample map tells us which input each sample corresponds to
    sample_map = batch.pop('overflow_to_sample_mapping')

    for i in range(len(batch['input_ids'])):
        # The sample index tells us which of the values in "samples" these features belong to
        sample_idx = sample_map[i]
        sequence_ids = batch.sequence_ids(i)

        # Add the ID to map these features back to the correct sample
        batch['example_id'].append(samples['id'][sample_idx])

        # Set offsets for non-context words to be None for ease of processing
        batch['offset_mapping'][i] = [o if sequence_ids[k] ==
                                      1 else None for k, o in enumerate(batch['offset_mapping'][i])]

    return batch


def collate_fn(inputs):
    '''
    Defines how to combine different samples in a batch
    '''
    input_ids = torch.tensor([i['input_ids'] for i in inputs])
    attention_mask = torch.tensor([i['attention_mask'] for i in inputs])
    labels = torch.tensor([i['labels'] for i in inputs])

    # Truncate to max length
    max_len = max(attention_mask.sum(-1))
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def val_collate_fn(inputs):
    input_ids = torch.tensor([i['input_ids'] for i in inputs])
    attention_mask = torch.tensor([i['attention_mask'] for i in inputs])

    # Truncate to max length
    max_len = max(attention_mask.sum(-1))
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]

    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def predict(model: nn.Module, valid_dl: DataLoader, device):
    """
    Evaluates the model on the given dataset
    :param model: The model under evaluation
    :param valid_dl: A `DataLoader` reading validation data
    :return: Predicted logits or probabilities for each example in the dataset
    """
    model.eval()
    all_logits = []

    # IMPORTANT: No gradient accumulation during prediction
    with torch.no_grad():
        for batch in tqdm(valid_dl, desc='Evaluation'):            
            logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            ).logits
            all_logits.append(logits)

        # Concatenate all the logits
        all_logits = torch.cat(all_logits, dim=0)

    return all_logits


def post_process_predictions(examples, dataset, logits, num_possible_answers=20, max_answer_length=30):
    all_start_logits, all_end_logits = logits
    # Build a map from example to its corresponding features. This will allow us to index from
    # sample ID to all of the features for that sample (in case they were split up due to long input)
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = defaultdict(list)
    for i, feature in enumerate(dataset):
        features_per_example[example_id_to_index[feature["example_id"]]].append(
            i)

    # Create somewhere to store our predictions
    predictions = OrderedDict()

    # Iterate through each sample in the dataset
    for j, sample in enumerate(tqdm(examples)):

        # Get the feature indices (all of the features split across the batch)
        feature_indices = features_per_example[j]
        # Get the original context which predumably has the answer text
        context = sample['document_plaintext']

        preds = []
        # Iterate through all of the features
        for ft_idx in feature_indices:

            # Get the start and end answer logits for this input
            start_logits = all_start_logits[ft_idx]
            end_logits = all_end_logits[ft_idx]

            # Get the offsets to map token indices to character indices
            offset_mapping = dataset[ft_idx]['offset_mapping']

            # Sort the logits and take the top N
            start_indices = np.argsort(start_logits)[
                ::-1][:num_possible_answers]
            end_indices = np.argsort(end_logits)[::-1][:num_possible_answers]

            # Iterate through start and end indices
            for start_index in start_indices:
                for end_index in end_indices:

                    # Ignore this combination if either the indices are not in the context
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping) or offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue

                    # Also ignore if the start index is greater than the end index of the number of tokens
                    # is greater than some specified threshold
                    if start_index > end_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    ans_text = context[offset_mapping[start_index]
                                       [0]:offset_mapping[end_index][1]]
                    preds.append({
                        'score': start_logits[start_index] + end_logits[end_index],
                        'text': ans_text
                    })

        if len(preds) > 0:
            # Sort by score to get the top answer
            answer = sorted(preds, key=lambda x: x['score'], reverse=True)[0]
        else:
            answer = {'score': 0.0, 'text': ""}

        predictions[sample['id']] = answer['text']
    return predictions
