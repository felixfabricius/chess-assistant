import polars as pl
import numpy as np
import json
import torch

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from pathlib import Path

from chess_assistant.model.config import TARGET_MAP

class squareDataset(Dataset):
    def __init__(self, transform, target_transform, csv_path: Path = Path("data/generated/data.csv"), split: str = "train"):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Split must be of type train, val or test. Got {split}.")
        self.data = pl.read_csv(csv_path).filter(pl.col("setup_split").eq("split"))
        self.transform = transform
        self.target_transform = target_transform

        # TODO: support transforms;
        # Transform images using torchvision v2 transforms ToImage, ToDtype etc. 
        # Transform targets into integer labels

    def __len__(self):
        return self.data.height

    def __getitem__(self, idx):
        square = self.data[idx, "square"]
        img_path = Path(self.data[idx, "square_image_path"]).parent / f"{square}_masked.npy"
            # TODO: remove this above workaround. Necessary at the moment because the
            # wrong image paths are saved in the csv. 
        image = np.load(img_path)

        if self.transform:
            image = self.transform(image)

        label = self.data[idx, "square_label"]
        if self.target_transform:
            label = self.target_transform(label)

        # what metadata do we want?
        # OHE version of which square is at the top; this can be accessed using setup_id -> setup calibration metadata 
        # then need to access square metadata; access using 
        metadata = []
        square_metadata_path = img_path.parent / f"{square}_metadata.json"
        with open(square_metadata_path, "r") as f:
            square_metadata = json.load(f)
            metadata.extend([square_metadata[key] for key in ["top", "left"]])
        setup_metadata_path = Path("data/generated") / self.data[idx, "setup_id"] / "calibration_metadata.json"
        with open(setup_metadata_path, "r") as f:
            setup_metadata = json.load(setup_metadata_path)
            metadata.extend([
                px_coordinate 
                for corner in ["a1", "a8", "h8", "h1"] 
                for px_coordinate in setup_metadata["actual_corners_px"][corner]
            ])
        metadata = torch.tensor(metadata)
        

        return image, metadata, label


def create_dataloader(split):
    
    
    dataset = squareDataset(
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), 
        target_transform=lambda piece: TARGET_MAP[piece],
            # alternatively, could simply write TARGET_MAP.__getitem__
            # also: could arguably do this directly in the dataset...
        csv_path=Path("data/generated/data.csv"),
    )
