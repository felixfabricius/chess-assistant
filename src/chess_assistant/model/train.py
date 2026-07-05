def train(model, dataloader, loss_fn, optimizer):
    for batch, (X, metadata, labels) in enumerate(dataloader):
        preds = model(X, metadata) # shape: (batch_size, 13)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()