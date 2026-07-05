import pytest
import torch
from torchvision.transforms import v2

from chess_assistant.model.data import create_dataloader, squareDataset
from chess_assistant.model.config import INVERSE_TARGET_MAP

@pytest.fixture
def dataset(scope="module"):
    return squareDataset()

def test_dataset_construction(dataset):
    assert isinstance(len(dataset), int) and len(dataset) > 0

def test_dataset_getitem(dataset):
    label = dataset[0][2]
    assert INVERSE_TARGET_MAP[label] == "R"

def test_metadata_shape(dataset):
    assert dataset[0][1].shape == (10,) # 5 coordinates: 4 board corners, and top left corner of square

### Test transformations
def test_transform(dataset):
    image = dataset[0][0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (4, 144, 144)
    assert image[3].max() == 1.0 # ensure mask channel is normalised correctly

def test_target_transform(dataset):
    label = dataset[0][2]
    assert isinstance(label, int)
    assert label in [i for i in range(13)]

### Test dataloader
@pytest.mark.parametrize("batch_size", [1, 4, 10, 64])
def test_dataloader(batch_size):
    dataloader = create_dataloader("train", batch_size)
    batch = next(iter(dataloader))
    assert len(batch) == 3
    assert batch[0].shape == (batch_size, 4, 144, 144)
    assert batch[1].shape == (batch_size, 10)
    assert batch[2].shape == (batch_size,)

### Test datatypes
def test_datatypes():
    dataloader = create_dataloader("train", 64)
    batch = next(iter(dataloader))
    assert batch[0].dtype == torch.float32
    assert batch[1].dtype == torch.float32
    assert batch[2].dtype == torch.long