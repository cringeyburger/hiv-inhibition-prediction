import torch
from sklearn.metrics import classification_report


def eval_model(device, model, test_loader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels.numpy())

    return classification_report(true_labels, predictions)
