import torch
import wandb
import copy
from torch import nn
from pathlib import Path
from omegaconf import DictConfig
import hydra
from dotenv import load_dotenv

from chess_assistant.model.model import SquareClassifier
from chess_assistant.model.data import create_dataloader
from chess_assistant.model.train import train
from chess_assistant.model.evaluate import evaluate

load_dotenv() # for api keys

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig):
    run_name = (
        f"{('[' + config.get('prefix') + '] | ') if config.get('prefix') else ''}"
        f"Model: {config.model}"
        f"{f' | {config.note}' if config.note else ''}"
    )
    run = wandb.init(
        project="chess-assistant",
        name=run_name,
        config=config,
        dir=".cache/wandb"
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
        batch_size=config.training.get("batch_size", 64),
        shuffle=True
    )
    val_dataloader = create_dataloader("val", batch_size=64, shuffle=False)

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

    epochs = config.training.get("epochs", 1)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}\n------------------------------")
        train_metrics = train(
            model=model, 
            dataloader=train_dataloader, 
            loss_fn=loss_fn, 
            optimizer=optimizer,
            debug=config.get("debug", False)
        )
        val_metrics = evaluate(
            model=model, 
            dataloader=val_dataloader, 
            loss_fn=loss_fn, 
            split="val", 
            csv_path=Path("data/generated/data.csv")
        )

        run.log({"epoch": epoch, **train_metrics, **val_metrics})

        # Update best model
        if val_metrics["eval/square/avg_loss"] < lowest_loss:
            lowest_loss = val_metrics["eval/square/avg_loss"]
            best_model = copy.deepcopy(model.state_dict())
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch

    # Persist the best version of the model
    cache_path = Path(".cache")
    cache_path.mkdir(exist_ok=True)
    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": best_model,
            "optimizer_state_dict": optimizer_state
        },
        cache_path / "model.pt"
    )
    artifact = wandb.Artifact(name=f"model_and_optimizer", type="model")
    artifact.add_file(".cache/model.pt")
    run.log_artifact(artifact)

    run.finish()

    # Load dataset
    return None 

if __name__ == "__main__":
    main()