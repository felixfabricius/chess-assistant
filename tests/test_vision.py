from chess_assistant.vision import SquareEstimate, BoardEstimate, BoardEstimator
from chess_assistant.model.model import SquareClassifier, SquareClassifierMultiHead
from chess_assistant.model.config import TARGET_MAP
from pathlib import Path

import math
import torch
import pytest

CALIBRATION_METADATA_PATH = Path("data/generated/2026-07-01_175334/calibration_metadata.json")
SQUARE_IMAGE_PATH = Path("data/generated/2026-07-01_175334/board_2026-07-01_175602/squares/a1/a1.png")
SQUARES_DIR = SQUARE_IMAGE_PATH.parent.parent

### Test that estimate_square and estimate_board work and return a valid board
@pytest.fixture
def board_estimator(scope="module"):
    return BoardEstimator(
        model_type="CNN", 
        calibration_metadata_path=CALIBRATION_METADATA_PATH,
        model=SquareClassifier()
    )

def test_estimate_square(board_estimator):
    square_estimate = board_estimator.estimate_square(SQUARE_IMAGE_PATH)
    assert isinstance(square_estimate.empty, float)
    assert isinstance(square_estimate, SquareEstimate)

def test_estimate_board(board_estimator):
    board_estimate = board_estimator.estimate_board(SQUARES_DIR)
    assert isinstance(board_estimate.a1, SquareEstimate)
    assert isinstance(board_estimate, BoardEstimate)


### The same, but with the multi-head model (model 3) through BoardEstimator
@pytest.fixture
def multihead_board_estimator(scope="module"):
    model = SquareClassifierMultiHead()
    # eval() is required: BatchNorm1d(10) errors on a batch-size-1 forward in train mode.
    model.eval()
    return BoardEstimator(
        model_type="CNN",
        calibration_metadata_path=CALIBRATION_METADATA_PATH,
        model=model
    )

def test_estimate_square_multihead(multihead_board_estimator):
    square_estimate = multihead_board_estimator.estimate_square(SQUARE_IMAGE_PATH)
    assert isinstance(square_estimate, SquareEstimate)
    # The 13 stored values are reconstructed log-probabilities; exp() forms a valid
    # distribution and softmax over them (as stated in the task) sums to ~1.
    values = torch.tensor([getattr(square_estimate, label) for label in TARGET_MAP])
    assert math.isclose(values.exp().sum().item(), 1.0, abs_tol=1e-4)
    assert math.isclose(torch.softmax(values, dim=-1).sum().item(), 1.0, abs_tol=1e-6)
