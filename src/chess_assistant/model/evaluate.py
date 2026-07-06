import torch
import polars as pl
import numpy as np

from pathlib import Path

from chess_assistant.vision import BoardEstimator
from chess_assistant.game import ChessGame

_split_data_cache = {}

def evaluate(model, dataloader, loss_fn, split, csv_path):
    ### Calculate:
        # losses for each individual datapoint;; perhaps averaged across batch
        # accuracy for each individual square
    n = len(dataloader.dataset)
    loss = 0
    n_correct = 0
    for (images, metadata, labels) in dataloader:
        # Items in batch
        n_batch = labels.shape[0]
        # Get loss
        with torch.no_grad():
            logits = model(images, metadata)
            loss += loss_fn(logits, labels).item() * n_batch
            # loss_fn returns the mean loss over items in batch; 
            # reason: default "reduction" of nn.CrossEntropyLoss is "mean"
            # therefore: sum this to be loss across all datapoints in batch, and then
            # divide by number of datapoints later
            # at this step, this would be identical to specifying CE loss with reduction = "mean"
        n_correct += (logits.argmax(dim=1) == labels).sum().item()
            # Original version was: torch.sum(torch.argmax(logits, 1) == labels).item()
            # But method access is a bit neater
    avg_loss = loss / n
    prop_correct = n_correct / n

    ### On position level
        # check if current position is valid - this should also be returned from dataloader
        # if it is; then return previous fen
        # then get the machinery going: predict all the square in one board position
        # note: this dataloader should therefore perhaps operate at this slightly higher level?
        # i.e. we always want to make predicitons for al the squares of a board - don't want to mix 
        # between boards.
    cache_key = (str(csv_path), split)
    if cache_key not in _split_data_cache:
        _split_data_cache[cache_key] = (
            pl.read_csv(csv_path)
            .filter(
                pl.col("setup_split").eq(split),
                pl.col("valid_game_position")
            )
        )
    data = _split_data_cache[cache_key]

    # For each of the unique positions, 
    board_position_ids = data.unique("image_id")["image_id"]

    correct_moves = 0
    correct_move_normalised_rank = []
    n_valid = 0

    for board_position_id in board_position_ids:
        # TODO: perhaps write test to check that .first() would yield same exact result
        # as .unique()
        board_position_data = (
            data.filter(pl.col("image_id").eq(board_position_id))
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
            model_type="CNN",
            model=model,
            calibration_metadata_path=Path("data/generated") / board_position_data["setup_id"] / "calibration_metadata.json"
        )
        board_estimator.estimate_board(squares_dir)

        game = ChessGame(fen=board_position_data["previous_board_fen"], model_type="CNN")
        assert game.board.is_valid() # should not arrive at an invalid position this way
        estimated_moves = game.estimate_move(board_estimator.board_estimate)

        if len(estimated_moves) == 0:
            continue

        # This returns list of moves
        if estimated_moves[0]["move"] == board_position_data["move_uci"]:
            correct_moves += 1

        for i, scored_move in enumerate(estimated_moves):
            if scored_move["move"] == board_position_data["move_uci"]:
                correct_move_normalised_rank.append(1 - i / len(estimated_moves)) 
                # TODO: should not divide by len(estimated_moves) - 1; should be
                # len(estimated_moves)
                continue
        
        n_valid += 1
    
    metrics = {
        "eval/square/n": n,
        "eval/square/avg_loss": avg_loss,
        "eval/square/prop_correct_square": prop_correct,
        "eval/board/n_valid": n_valid,
        "eval/board/prop_correct_board": correct_moves / n_valid if n_valid > 0 else None,
        "eval/board/correct_normalised_rank": np.mean(correct_move_normalised_rank)
    }

    return metrics