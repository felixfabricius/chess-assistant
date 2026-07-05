import pytest
import torch
import time
from chess_assistant.model.data import create_dataloader
from chess_assistant.model.model import SquareClassifier

###
@pytest.fixture
def dataloader(scope="module"):
    return create_dataloader("train", 64)

@pytest.fixture
def model(scope="module"):
    return SquareClassifier()

def test_forward_pass(dataloader, model):
    start = time.perf_counter()
    batch = next(iter(dataloader))
    output = model(batch[0], batch[1])
    assert output.shape == (64, 13)
    assert output.dtype == torch.float32
    end = time.perf_counter()
    print(f"{end - start:.6f}")
