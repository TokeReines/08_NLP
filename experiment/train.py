
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score

def train(
    model,
    optimizer,
    scheduler,
    dataset,
    validation_set,
    epochs=5,
    batch_size=32,
    lr=0.001,
    device="cuda",
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    model.to(device)
    best_model = None
    best_accuracy = None
    for epoch in range(epochs):
        epoch_loss = None
        model.train()
        for q_ids, q_mask, d_ids, d_mask, labels in tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{epochs}"
        ):
            q_ids, q_mask, d_ids, d_mask, labels = (
                q_ids.to(device).squeeze(1),
                q_mask.to(device).squeeze(1),
                d_ids.to(device).squeeze(1),
                d_mask.to(device).squeeze(1),
                labels.to(device),
            )

            optimizer.zero_grad()
            outputs = model(q_ids, q_mask, d_ids, d_mask).squeeze()
            loss = criterion(outputs, labels.float())

            loss.backward()

            if epoch_loss is None:
                epoch_loss = loss.item()
            else:
                epoch_loss += loss.item()

            optimizer.step()
            scheduler.step()

        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

        accuracy = evaluate(model, validation_set, batch_size=batch_size, device=device)
        if best_accuracy is None or accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict()
            
    return best_model


def evaluate(model, dataset, batch_size=32, device="cuda"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for q_ids, q_mask, d_ids, d_mask, labels in tqdm(dataloader):
        q_ids, q_mask, d_ids, d_mask, labels = (
            q_ids.to(device).squeeze(1),
            q_mask.to(device).squeeze(1),
            d_ids.to(device).squeeze(1),
            d_mask.to(device).squeeze(1),
            labels.to(device),
        )

        outputs = model(q_ids, q_mask, d_ids, d_mask).squeeze()

        predicted_labels = (outputs > 0.5).int()
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)

    total_loss /= len(dataloader)
    accuracy = correct_predictions / total_samples
    print(f"Loss: {total_loss}, Accuracy: {accuracy:.2%}")
    return accuracy
    # f1 = f1_score(labels.cpu(), predicted_labels.cpu())
    # return 
