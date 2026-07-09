import pytest
import torch
import time
from chess_assistant.model.data import create_dataloader
from chess_assistant.model.model import SquareClassifier, SquareClassifierMultiHead

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

@pytest.fixture
def multihead_model(scope="module"):
    return SquareClassifierMultiHead()

def test_forward_pass_multihead(dataloader, multihead_model):
    batch = next(iter(dataloader))
    logit_empty, logits_color, logits_type = multihead_model(batch[0], batch[1])
    assert logit_empty.shape == (64,)
    assert logits_color.shape == (64, 2)
    assert logits_type.shape == (64, 6)
    assert logit_empty.dtype == torch.float32
    assert logits_color.dtype == torch.float32
    assert logits_type.dtype == torch.float32
