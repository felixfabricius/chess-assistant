from chess_assistant.vision import SquareEstimate, BoardEstimate, BoardEstimator
from chess_assistant.model.model import SquareClassifier
from pathlib import Path

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
