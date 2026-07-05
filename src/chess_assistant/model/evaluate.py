import polars as pl

from pathlib import Path

from chess_assistant.vision import BoardEstimator
from chess_assistant.game import ChessGame

def evaluate(model, loss_fn, split, csv_path):
    ### Calculate:
        # losses for each individual datapoint;; perhaps averaged across batch
        # accuracy for each individual square
    
    ### On position level
        # check if current position is valid - this should also be returned from dataloader
        # if it is; then return previous fen
        # then get the machinery going: predict all the square in one board position
        # note: this dataloader should therefore perhaps operate at this slightly higher level?
        # i.e. we always want to make predicitons for al the squares of a board - don't want to mix 
        # between boards.
    data = (
        pl.read_csv(csv_path)
        .filter(
            pl.col("setup_split").eq(split),
            pl.col("valid_game_position")
        )
    )

    # For each of the unique positions, 
    board_position_ids = data.unique("image_id")["image_id"]

    for board_position_id in board_position_ids:
        # TODO: perhaps write test to check that .first() would yield same exact result
        # as .unique()
        board_position_data = (
            data.filter(pl.col("image_id").eq("board_position_id"))
            .select(
                pl.col("setup_id").first(),
                pl.col("previous_board_fen").first(),
                pl.col("board_fen").first(),
                pl.col("move_uci").first()
            )
        )
        board_position_data = board_position_data.to_dicts()[0]
        squares_dir = Path("data/generated") / board_position_data["setup_id"] / board_position_id / "squares"
        
        board_estimator = BoardEstimator(
            model_type="CNN"
            model=model,
            calibration_metadata_path=Path("/data/generation") / board_position_data["setup_id"] / "calibration_metadata.json"
        )
        board_estimator.estimate_board(squares_dir)

        game = ChessGame(fen=board_position_data["previous_board_fen"], model_type="CNN")
        assert game.board.is_valid() # should not arrive at an invalid position this way
        game.estimate_move(board_estimator.board_estimate)

        # This returns list of moves
        
                # Find the unique squares_directory associated with that position
        # Also find the unique previous FEN
        # And find the current move
            # TODO: perhaps rewrite so this doesn't require a config?
