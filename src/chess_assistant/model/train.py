def train(model, dataloader, loss_fn, optimizer):
    n_batches = len(dataloader)
    loss_logging_threshold = n_batches * 0.8 // 1
    n_loss_samples = 0
    total_loss = 0
    for batch, (X, metadata, labels) in enumerate(dataloader):
        preds = model(X, metadata) # shape: (batch_size, 13)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch + 1 >= loss_logging_threshold:
            n_batch = labels.shape[0]
            total_loss += loss.item() * n_batch # multiply by n_batch since reduction="mean"
            n_loss_samples += n_batch
    
    return {
        "train/square/recent_loss": loss / n_loss_samples,
        "train/square/n_recent_loss": n_loss_samples,
    }