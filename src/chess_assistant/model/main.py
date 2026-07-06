import torch
import wandb
import copy
from torch import nn
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv

from chess_assistant.model.model import SquareClassifier
from chess_assistant.model.data import create_dataloader
from chess_assistant.model.train import train
from chess_assistant.model.evaluate import evaluate

load_dotenv() # for api keys

def main(config_path: Path = "src/chess_assistant/model/config.yaml"):
    config = OmegaConf.load(config_path)

    run_name = (
        f"Model: {config.model}"
        f"{f' | {config.note}' if config.note else ''}"
    )
    run = wandb.init(
        project="chess-assistant",
        name=run_name,
        config=config
    )
    run.define_metric("epoch") # tells wandb I will log metric
    run.define_metric("*", step_metric="epoch") # for all other metrics matching "*", use epoch as x-axis

    model = SquareClassifier()
    lowest_loss = float("inf")
    best_model = None
    optimizer_state = None
    best_epoch = 0
    
    train_dataloader = create_dataloader(
        split="train",
        batch_size=config.get("batch_size", 64),
        shuffle=True
    )
    val_dataloader = create_dataloader("val", 64, False)

    # Hyperparameters
    lr = config.optimizer.lr
    weight_decay = config.optimizer.get("weight_decay", 1e-4)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    epochs = config.get("epochs", 1)
    for epoch in range(epochs):
        train_metrics = train(model, train_dataloader, loss_fn, optimizer)
        val_metrics = evaluate(model, val_dataloader, loss_fn)

        run.log({"epoch": epoch, **train_metrics, **val_metrics})

        # Update best model
        if val_metrics["eval/square/avg_loss"] < lowest_loss:
            lowest_loss = val_metrics["eval/square/avg_loss"]
            best_model = copy.deepcopy(model.state_dict())
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch

    # Persist the best version of the model
    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": best_model.state_dict(),
            "optimizer_state_dict": optimizer_state
        }
        , "model.pt"
    )
    artifact = wandb.Artifact(name=f"model_and_optimizer", type="model")
    artifact.add_file("model.pt")
    run.log_artifact(artifact)

    run.finish()

    # Load dataset
    return None 

if __name__ == "__main__":
    main()