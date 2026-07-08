def train(model, dataloader, loss_fn, optimizer, debug, device):
    model.train()
    n_batches = len(dataloader)
    loss_logging_threshold = n_batches * 0.8 // 1
    n_loss_samples = 0
    total_loss = 0
    for batch, (X, metadata, labels) in enumerate(dataloader):
        X, metadata, labels = X.to(device, non_blocking=True), metadata.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        preds = model(X, metadata) # shape: (batch_size, 13)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if debug and batch > 3:
            break

        if batch % 10 == 0:
            print(f"Batch {batch + 1:>4d} / {n_batches:>4d} | Loss: {loss.item():.2f}")

        if batch + 1 >= loss_logging_threshold:
            n_batch = labels.shape[0]
            total_loss += loss.item() * n_batch # multiply by n_batch since reduction="mean"
            n_loss_samples += n_batch
    
    return {
        "train/square/recent_loss": total_loss / n_loss_samples if not debug else 0, # can get DivByZero Error if debugging
        "train/square/n_recent_loss": n_loss_samples,
    }