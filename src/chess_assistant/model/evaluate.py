import torch
import torch.nn.functional as F
import polars as pl
import numpy as np
import wandb

from pathlib import Path

from chess_assistant.vision import BoardEstimator
from chess_assistant.game import ChessGame
from chess_assistant.model.config import (
    INVERSE_TARGET_MAP,
    INVERSE_COLOR_MAP,
    INVERSE_TYPE_MAP,
    IGNORE_INDEX,
    reconstruct_13way_logprobs,
)

_split_data_cache = {}

def evaluate(model, dataloader, loss_fns, loss_weights, split, csv_path, device):
    model.eval() # important for batch norm; want to use mean and sd that were accumulated during training
    ### Calculate:
        # per-head losses (empty / color / type) and their confusion matrices
        # the reconstructed 13-way square loss + accuracy + confusion matrix (kept for
        #   comparability with the single-head models 1 & 2 - drives checkpoint selection)
    n = len(dataloader.dataset)

    empty_loss_sum = 0.0
    color_loss_sum = 0.0
    type_loss_sum = 0.0
    square_loss_sum = 0.0  # reconstructed 13-way CrossEntropy
    n_nonempty = 0         # number of non-empty rows actually seen (denominator for color/type)
    n_correct_square = 0

    # per-head predictions/targets, collected across the eval set for the confusion matrices
    empty_preds, empty_targets = [], []
    color_preds, color_targets = [], []
    type_preds, type_targets = [], []
    all_preds13, all_labels13 = [], []

    for images, metadata, is_piece, color_target, type_target in dataloader:
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        # is_piece comes off the default collate as float64; BCEWithLogitsLoss needs the
        # target dtype to match the float32 logits.
        is_piece = is_piece.to(device, non_blocking=True).float()
        color_target = color_target.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)
        n_batch = is_piece.shape[0]

        # exactly the non-empty rows (color/type are IGNORE_INDEX iff empty)
        nonempty_mask = color_target != IGNORE_INDEX
        n_nonempty_batch = int(nonempty_mask.sum().item())

        with torch.no_grad():
            logit_empty, logits_color, logits_type = model(images, metadata)

            # ---- per-head losses ----
            # empty loss is over every row -> normalise by total n
            empty_loss_sum += loss_fns["empty"](logit_empty, is_piece).item() * n_batch
            # color/type CrossEntropyLoss(reduction="mean", ignore_index=...) averages over the
            # non-empty rows only; scale by that count and divide by total non-empty later.
            # Guard against an all-empty batch (mean over zero rows would be NaN).
            if n_nonempty_batch > 0:
                color_loss_sum += loss_fns["color"](logits_color, color_target).item() * n_nonempty_batch
                type_loss_sum += loss_fns["type"](logits_type, type_target).item() * n_nonempty_batch
                n_nonempty += n_nonempty_batch

            # ---- reconstructed 13-way view (kept, comparable to models 1 & 2) ----
            logprobs = reconstruct_13way_logprobs(logit_empty, logits_color, logits_type)  # (batch, 13)
            # Re-derive the 13-way integer label from the decomposed targets (the dataset no
            # longer returns it): empty -> 0; white piece -> type+1; black piece -> type+7.
            label13 = torch.zeros_like(type_target)
            label13[nonempty_mask] = type_target[nonempty_mask] + 1 + color_target[nonempty_mask] * 6
            square_loss_sum += F.cross_entropy(logprobs, label13).item() * n_batch
            preds13 = logprobs.argmax(dim=-1)
            n_correct_square += (preds13 == label13).sum().item()

        # ---- collect predictions/targets for the confusion matrices ----
        empty_preds.append((torch.sigmoid(logit_empty) > 0.5).long())
        empty_targets.append(is_piece.long())
        color_preds.append(logits_color.argmax(dim=-1)[nonempty_mask])
        color_targets.append(color_target[nonempty_mask])
        type_preds.append(logits_type.argmax(dim=-1)[nonempty_mask])
        type_targets.append(type_target[nonempty_mask])
        all_preds13.append(preds13)
        all_labels13.append(label13)

    empty_avg = empty_loss_sum / n
    color_avg = color_loss_sum / n_nonempty if n_nonempty > 0 else 0.0
    type_avg = type_loss_sum / n_nonempty if n_nonempty > 0 else 0.0
    square_avg = square_loss_sum / n  # reconstructed 13-way CE -> checkpoint metric
    total_avg = (
        loss_weights["empty"] * empty_avg
        + loss_weights["color"] * color_avg
        + loss_weights["type"] * type_avg
    )
    prop_correct = n_correct_square / n

    # W&B's confusion-matrix chart sorts axis labels alphabetically, ignoring table row
    # order; zero-padded index prefixes make the alphabetical order coincide with the
    # intended (TARGET_MAP / COLOR_MAP / TYPE_MAP) order.
    def _confusion_matrix(targets, preds, class_names):
        return wandb.plot.confusion_matrix(
            y_true=torch.cat(targets).to("cpu").numpy(),
            preds=torch.cat(preds).to("cpu").numpy(),
            class_names=class_names,
        )

    confusion_matrix_plot = _confusion_matrix(
        all_labels13, all_preds13, [f"{i:02d}_{INVERSE_TARGET_MAP[i]}" for i in range(13)]
    )
    empty_confusion_matrix = _confusion_matrix(
        empty_targets, empty_preds, ["00_empty", "01_piece"]
    )
    color_confusion_matrix = _confusion_matrix(
        color_targets, color_preds, [f"{i:02d}_{INVERSE_COLOR_MAP[i]}" for i in range(2)]
    )
    type_confusion_matrix = _confusion_matrix(
        type_targets, type_preds, [f"{i:02d}_{INVERSE_TYPE_MAP[i]}" for i in range(6)]
    )

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
            calibration_metadata_path=Path("data/generated") / board_position_data["setup_id"] / "calibration_metadata.json",
            device=device
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
        "eval/square/avg_loss": square_avg,  # reconstructed 13-way CE (drives checkpoint selection)
        "eval/square/prop_correct_square": prop_correct,
        "eval/square/confusion_matrix": confusion_matrix_plot,
        "eval/empty/avg_loss": empty_avg,
        "eval/empty/confusion_matrix": empty_confusion_matrix,
        "eval/color/avg_loss": color_avg,
        "eval/color/confusion_matrix": color_confusion_matrix,
        "eval/type/avg_loss": type_avg,
        "eval/type/confusion_matrix": type_confusion_matrix,
        "eval/total/avg_loss": total_avg,  # weighted combined loss (diagnostic, mirrors train/total)
        "eval/board/n_valid": n_valid,
        "eval/board/prop_correct_board": correct_moves / n_valid if n_valid > 0 else None,
        "eval/board/correct_normalised_rank": np.mean(correct_move_normalised_rank),
    }

    return metrics
