import torch
from torch import nn
from pathlib import Path
from omegaconf import OmegaConf

from chess_assistant.model.model import SquareClassifier
from chess_assistant.model.data import create_dataloader
from chess_assistant.model.train import train
from chess_assistant.model.evaluate import evaluate

def train(config_path: Path = "src/chess_assistant/model/config.yaml"):
    config = OmegaConf.load(config_path)

    model = SquareClassifier()    
    train_dataloader = create_dataloader(
        split="train",
        batch_size=config.get("batch_size", 64)
    )

    # Hyperparameters
    lr = config.optimizer.lr
    weight_decay = config.optimizer.get("weight_decay", 1e-4)

    # Loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    epochs = config.get("epochs", 1)
    for epoch in epochs:
        train(model, train_dataloader, loss, optimizer)
        

    # Load dataset
    return None 

if __name__ == "__main__":
    train()