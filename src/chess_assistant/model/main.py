import torch
import wandb
import copy
import numpy as np
import polars as pl
from torch import nn
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from dotenv import load_dotenv

from chess_assistant.model.model import SquareClassifier, SquareClassifier2
from chess_assistant.model.data import create_dataloader
from chess_assistant.model.train import train
from chess_assistant.model.evaluate import evaluate
from chess_assistant.model.config import TARGET_MAP

load_dotenv() # for api keys

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.data.weighting == "inverse_root" and config.note == "":
        config.note = "Weighting: Inverse Root"
    if config.get("debug") and not config.get("prefix"):
        # Standard version might not work because config is in "struct mode", which
        # is meant to prevent actual adding of keys to config
        # config.prefix = "[test run]"
        OmegaConf.update(config, "prefix", "test run", force_add=True)
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

    assert config.model in [1, 2]
    model = SquareClassifier() if config.model == 1 else SquareClassifier2()
    model = model.to(device) # model.to(device) is also in place so would be sufficient
    lowest_loss = float("inf")
    best_model_state = None
    optimizer_state = None
    best_epoch = 0
    
    dataloader_cfg = config.get("data", {}).get("dataloader", {})
    train_dataloader = create_dataloader(
        split="train",
        shuffle=True,
        batch_size=config.training.get("batch_size", 64),
        num_workers=dataloader_cfg.get("num_workers", 0),
        persistent_workers=dataloader_cfg.get("persistent_workers", False),
        pin_memory=dataloader_cfg.get("pin_memory", False) and device == torch.device("cuda") 
    )
    val_dataloader = create_dataloader(
        split="val",
        shuffle=False,
        batch_size=config.training.get("batch_size", 64),
        num_workers=dataloader_cfg.get("num_workers", 0),
        persistent_workers=dataloader_cfg.get("persistent_workers", False),
        pin_memory=dataloader_cfg.get("pin_memory", False) and device == torch.device("cuda") 
    )

    # Hyperparameters
    lr = config.optimizer.lr
    weight_decay = config.optimizer.get("weight_decay", 1e-4)

    # 
    if config.data.get("weighting") == "inverse_root":
        weights = torch.zeros(13, dtype=torch.float32).to(device)
        csv_path = Path(config.data.get("csv_path"))
        assert csv_path.exists()
        data = pl.read_csv(csv_path).filter(pl.col("setup_split").eq("train"))
        counts = data["label"].value_counts()
        for row in counts.iter_rows(named=True):
            weights[TARGET_MAP[row["label"]]] = 1 / np.sqrt(row["count"])
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()
    eval_loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    epochs = config.training.get("epochs", 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config.get("scheduler", {}).get("eta_min", 1e-6)
    )
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}\n------------------------------")
        train_metrics = train(
            model=model, 
            dataloader=train_dataloader, 
            loss_fn=loss_fn, 
            optimizer=optimizer,
            debug=config.get("debug", False),
            device=device
        )
        val_metrics = evaluate(
            model=model, 
            dataloader=val_dataloader, 
            loss_fn=eval_loss_fn, 
            split="val", 
            csv_path=Path("data/generated/data.csv"),
            device=device
        )

        run.log({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **train_metrics, **val_metrics})

        # Update best model
        if val_metrics["eval/square/avg_loss"] < lowest_loss:
            lowest_loss = val_metrics["eval/square/avg_loss"]
            best_model_state = copy.deepcopy(model.state_dict())
            optimizer_state = copy.deepcopy(optimizer.state_dict())
                # Reason for .to("cpu") here: if device is CUDA, then state_dict()
                # and the AdamW momentum buffers in optimizer.state_dict()
                # contain CUDA tensors, which get pickled with their device ifo.
                # So if in a future script I try to load this checkpoint into my
                # GPU-less machine, plain torch.load(cache_path) would crash.
                # Can pass map_location="cpu" to torch.load - THIS WOULD ACTUALLY BE EASIEST.
                # Also: don't want to write 'best_model = copy.deepcopy(model.to("cpu").state_dict())'
                # because this would move model in place
            best_epoch = epoch

        scheduler.step()

    # Persist the best version of the model
    cache_path = Path(".cache") / f"model_{run.id}.pt"
    cache_path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": best_model_state,
            "optimizer_state_dict": optimizer_state
        },
        cache_path
    )
    artifact = wandb.Artifact(name=f"model_and_optimizer", type="model")
    artifact.add_file(cache_path)
    run.log_artifact(artifact)

    run.finish()

    return None 

if __name__ == "__main__":
    main()