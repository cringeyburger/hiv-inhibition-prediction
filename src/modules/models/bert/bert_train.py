def train_epoch(device, model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


def train_model(device, model, epochs, train_loader, optimizer, criterion):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_epoch(device, model, train_loader, optimizer, criterion)
        print(f"Training Loss: {train_loss:.4f}")
