import torch
from transformers import BertForSequenceClassification


def get_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    return model


def get_optimizer_with_llrd(model, base_lr=5e-5):
    optimizer_grouped_parameters = []

    for i, layer in enumerate(model.bert.encoder.layer):
        # Decay by 0.95 per layer
        lr = base_lr * (0.95 ** (len(model.bert.encoder.layer) - i))
        optimizer_grouped_parameters.append({"params": layer.parameters(), "lr": lr})

    optimizer_grouped_parameters.append(
        {"params": model.classifier.parameters(), "lr": base_lr}
    )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    return optimizer
