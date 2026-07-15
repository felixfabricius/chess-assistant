"""Shape/dtype smoke tests for the model forward passes, on a real batch off the training
dataloader. Needs the generated dataset (data/generated/data.csv) on disk.
"""

import pytest
import torch
import time
from chess_assistant.model.data import create_dataloader
from chess_assistant.model.model import SquareClassifier, SquareClassifierMultiHead


@pytest.fixture(scope="module")
def dataloader():
    return create_dataloader("train", shuffle=True, batch_size=64)

@pytest.fixture(scope="module")
def model():
    return SquareClassifier()

def test_forward_pass(dataloader, model):
    # Model 1: a single 13-way logit vector per square.
    start = time.perf_counter()
    batch = next(iter(dataloader))
    output = model(batch[0], batch[1])
    assert output.shape == (64, 13)
    assert output.dtype == torch.float32
    end = time.perf_counter()
    print(f"Time taken for batched forward pass with 64 squares: {end - start:.6f}")

@pytest.fixture(scope="module")
def multihead_model():
    return SquareClassifierMultiHead()

def test_forward_pass_multihead(dataloader, multihead_model):
    # Model 3: the three factored heads, one empty logit + 2 color + 6 type.
    batch = next(iter(dataloader))
    logit_empty, logits_color, logits_type = multihead_model(batch[0], batch[1])
    assert logit_empty.shape == (64,)
    assert logits_color.shape == (64, 2)
    assert logits_type.shape == (64, 6)
    assert logit_empty.dtype == torch.float32
    assert logits_color.dtype == torch.float32
    assert logits_type.dtype == torch.float32
