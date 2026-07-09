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
    Numerically stable (logsigmoid/log_softmax, no epsilon-clamping needed). Safe to feed
    directly into nn.CrossEntropyLoss or argmax, exactly like the old single-head logits.
    """
    log_p_empty = F.logsigmoid(logit_empty)            # log P(empty)
    log_p_nonempty = F.logsigmoid(-logit_empty)        # log(1 - P(empty))
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
