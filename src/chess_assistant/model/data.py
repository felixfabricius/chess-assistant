import polars as pl
import numpy as np
import json

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from pathlib import Path, PureWindowsPath

from chess_assistant.model.config import TARGET_MAP

TRAIN_TRANSFORM = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.03),
    v2.GaussianBlur(kernel_size=5, sigma=(1e-3, 1)), # kernel_size 5, so this is wider than the conv kernels
    v2.GaussianNoise(mean=0.0, sigma=0.02),
    v2.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.93, 1.07))
])

EVAL_TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

class squareDataset(Dataset):
    def __init__(
        self, 
        csv_path: Path = Path("data/generated/data.csv"), 
        split: str = "train",
        train_transform= v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), 
        eval_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        target_transform = TARGET_MAP.__getitem__,
    ):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Split must be of type train, val or test. Got {split}.")
        self.data = pl.read_csv(csv_path).filter(pl.col("setup_split").eq(split))
        self.split = split
        self.transform = train_transform if split == "train" else eval_transform
        self.target_transform = target_transform
        self.setup_metadata_store = {}

        # TODO: support transforms;
        # Transform images using torchvision v2 transforms ToImage, ToDtype etc. 
        # Transform targets into integer labels

    def __len__(self):
        return self.data.height

    def __getitem__(self, idx):
        square = self.data[idx, "square"]
        # square_image_path in the csv was written with str(Path(...)) on whichever OS
        # generated the data (currently Windows), so it may contain "\" separators.
        # Parse it as a Windows path first so it resolves correctly on Linux too.
        raw_path = PureWindowsPath(self.data[idx, "square_image_path"]).as_posix()
        # The CSV stores the annotated-image path; the model input is the 4-channel masked
        # array saved alongside it in the same per-square directory.
        img_path = Path(raw_path).parent / f"{square}_masked.npy"
            # TODO: not sure this way of accessing polars row item is robust to having no rows with that split.
        image = np.load(img_path)

        if self.transform:
            # TODO: reomve this workaroud. only reason I can't simply
            # self.transform(image) is that my mask values are currently
            # in {0, 1} rather than {0, 255}, so scale would make them too
            # small
            rgb = image[..., :3]
            mask = tv_tensors.Mask(image[..., 3]) 
                # CAREFUL: can't do .unsqueeze(dim=0) operation here; 
                # reason: this would downgrade the mask to a normal tensor

            # Note that the same transform object transforms differently each time
            transformed_rgb = self.transform(rgb)
            transformed_mask = self.transform(mask).unsqueeze(dim=0) # only unsqueeze now!!
                # the transforms that we don't want this to impact will not impact this!
                # e.g. v2.ToImage(), v2.GaussianNoise(...), ...
            
            image = torch.cat([transformed_rgb, transformed_mask], dim=0)
            # Note that shape is now (4, H, W) (rather than (H, W, 4))
            # Note also that type(transformed_image) is torch.Tensor
            # rather than a torchvision image (which is a subclass of Tensor)
            # and which can be useful for more advanced image transformations
        if not isinstance(image, torch.Tensor):
            raise TypeError("images must be torch tensors. Pass transform to ensure.")

        label = self.data[idx, "label"]
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
        
        setup_id = self.data[idx, "setup_id"]
        if setup_id not in self.setup_metadata_store:
            setup_metadata_path = Path("data/generated") / setup_id / "calibration_metadata.json"
            with open(setup_metadata_path, "r") as f:
                setup_metadata = json.load(f)
            self.setup_metadata_store[setup_id] = [
                px_coordinate 
                for corner in ["a1", "a8", "h8", "h1"] 
                for px_coordinate in setup_metadata["actual_corners_px"][corner]
            ]
        metadata.extend(self.setup_metadata_store[setup_id])
        metadata = torch.tensor(metadata, dtype=torch.float32)
        
        return image, metadata, label

        # TODO: for val/test we may also want image_id (to test if equal for all); valid_game_position; previous_board_fen; board_fen; move_uci
        # Issue with the dataloader approach to "randomly" get just images from one board position in one batch
        # As soon as just one row is removed from the CSV, this might no longer work
        # Alternative approach: load data.csv; then get some mask for the 
        # valid game positions in the current split;
        # Then for each of those board positions, perhaps call BoardEstimator with estimate_board.
        # (we already have the square_dir)
        # Perhaps need to reinitialise each each time with the previous FEN.
        # I think this second approach is neater!


def create_dataloader(
    split: str,
    shuffle: bool = False,
    batch_size: int = 64,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    train_transform = TRAIN_TRANSFORM, 
    eval_transform = EVAL_TRANSFORM,
    target_transform = TARGET_MAP.__getitem__,
    csv_path = Path("data/generated/data.csv"),
):  
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Split must be of type train, val or test. Got {split}.")
    dataset = squareDataset(
        csv_path=csv_path, 
        split=split, 
        train_transform=train_transform,
        eval_transform=eval_transform, 
        target_transform=target_transform
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory
    )

    return dataloader
