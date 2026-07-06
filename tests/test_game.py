import pytest

from chess_assistant.game import ChessGame
from chess_assistant.vision import SquareEstimate, BoardEstimate
from chess_assistant.config import SQUARES
from chess_assistant.model.config import TARGET_MAP

def create_board_estimate(square_occupants: dict[str, str]):
    """
    Creates a board where by default all squares are equally likely to be 
    occupied by any figure.
    Then modifies this board so that the squares in square_occupants are 
    occupied by the corresponding value.
    """
    for piece in square_occupants.values():
        assert piece in TARGET_MAP.keys() # "empty", "K", "Q", ...

    board_estimate = BoardEstimate()
    for square in SQUARES:
        if square not in square_occupants.keys():
            setattr(board_estimate, square, SquareEstimate())
        else:
            square_estimate = SquareEstimate()
            setattr(square_estimate, square_occupants[square], 100)
            setattr(
                board_estimate, 
                square,
                square_estimate
            )
    return board_estimate

### Test that estimate move works
@pytest.mark.parametrize(
    "move_uci, square_occupants",
    [
        ("e2e4", {"e2": "empty", "e4": "P"}),
        ("b1c3", {"b1": "empty", "c3": "N"})
    ]
)
def test_move_estimation(move_uci, square_occupants):
    game = ChessGame(model_type="CNN")
    board_estimate = create_board_estimate(square_occupants)
    move_estimate = game.estimate_move(board_estimate)
    assert move_estimate[0]["move"] == move_uci
