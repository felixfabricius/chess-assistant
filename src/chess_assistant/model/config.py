"""Label and metadata encodings shared by training (model/data.py, model/train.py,
model/evaluate.py) and inference (vision.py), plus the two functions that convert between the
13-way label space and the multi-head model's factored one. Anything both sides need to agree on
lives here, so the two encodings cannot drift apart.
"""

import torch
import torch.nn.functional as F

TARGET_MAP = {piece: label for label, piece in enumerate([
    "empty",
    "K", "Q", "R", "B", "N", "P",
    "k", "q", "r", "b", "n", "p"
])}

INVERSE_TARGET_MAP = {label: piece for piece, label in TARGET_MAP.items()}

# matches torch.nn.CrossEntropyLoss default ignore_index
IGNORE_INDEX = -100

COLOR_MAP = {"white": 0, "black": 1}
INVERSE_COLOR_MAP = {v: k for k, v in COLOR_MAP.items()}

TYPE_MAP = {"K": 0, "Q": 1, "R": 2, "B": 3, "N": 4, "P": 5}
INVERSE_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}

# One-hot index for which board corner is top-left in the camera image
# (calibration_metadata["camera_natural_orientation"]["order"]["tl"]). This is the model's
# only metadata now. Shared between training (model/data.py) and inference (vision.py) so the
# two encodings can never drift apart.
TOP_LEFT_OHE_MAP = {"a8": 0, "a1": 1, "h1": 2, "h8": 3}


def decompose_label(label: str) -> tuple[float, int, int]:
    """
    Decompose a 13-way label (as in TARGET_MAP) into:
      - is_piece: 1.0 if occupied, 0.0 if empty
      - color_target: 0 (white) / 1 (black), or IGNORE_INDEX if empty
      - type_target: 0..5 (K/Q/R/B/N/P), or IGNORE_INDEX if empty
    """
    if label == "empty":
        return 0.0, IGNORE_INDEX, IGNORE_INDEX
    color_target = COLOR_MAP["white"] if label.isupper() else COLOR_MAP["black"]
    type_target = TYPE_MAP[label.upper()]
    return 1.0, color_target, type_target


def reconstruct_13way_logprobs(logit_empty, logits_color, logits_type):
    """
    Combine the three heads into log-probabilities over the original 13-way TARGET_MAP
    labels, under a conditional-independence assumption between color and type given
    non-empty. Shape: (..., 13), indexed per TARGET_MAP / INVERSE_TARGET_MAP.

    Everything stays in log space via logsigmoid / log_softmax, so this is numerically stable
    and needs no epsilon-clamping. The result is a normalised log-probability vector, and since
    softmax(log p) == p it is a drop-in replacement for the old single-head logits: feeding it to
    argmax, to nn.CrossEntropyLoss, or to the softmax that game.py re-applies downstream all give
    the intended answer.
    """
    # The empty head is trained with BCEWithLogitsLoss against `is_piece` (1 = piece), so
    # sigmoid(logit_empty) == P(piece). `logit_empty` is therefore a piece-logit despite its
    # name, and the empty/non-empty branches must be assigned accordingly.
    log_p_nonempty = F.logsigmoid(logit_empty)         # log P(non-empty) = log sigmoid(logit_empty) = log P(piece)
    log_p_empty = F.logsigmoid(-logit_empty)           # log P(empty)     = log(1 - P(piece))
    log_p_color = F.log_softmax(logits_color, dim=-1)  # (..., 2)
    log_p_type = F.log_softmax(logits_type, dim=-1)    # (..., 6)

    out = torch.empty(logit_empty.shape + (13,), dtype=logit_empty.dtype, device=logit_empty.device)
    out[..., TARGET_MAP["empty"]] = log_p_empty
    for label, idx in TARGET_MAP.items():
        if label == "empty":
            continue
        color_idx = COLOR_MAP["white"] if label.isupper() else COLOR_MAP["black"]
        type_idx = TYPE_MAP[label.upper()]
        out[..., idx] = log_p_nonempty + log_p_color[..., color_idx] + log_p_type[..., type_idx]
    return out
