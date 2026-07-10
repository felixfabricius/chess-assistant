import torch
from torch import nn

from chess_assistant.model.config import (
    TARGET_MAP,
    IGNORE_INDEX,
    decompose_label,
    reconstruct_13way_logprobs,
)


### decompose_label
def test_decompose_label_empty():
    assert decompose_label("empty") == (0.0, IGNORE_INDEX, IGNORE_INDEX)

def test_decompose_label_white_piece():
    # "R" -> white rook: is_piece 1.0, color 0 (white), type 2 (R)
    assert decompose_label("R") == (1.0, 0, 2)

def test_decompose_label_black_piece():
    # "n" -> black knight: is_piece 1.0, color 1 (black), type 4 (N)
    assert decompose_label("n") == (1.0, 1, 4)


### reconstruct_13way_logprobs
def test_reconstruct_sums_to_one():
    torch.manual_seed(0)
    logit_empty = torch.randn(5)
    logits_color = torch.randn(5, 2)
    logits_type = torch.randn(5, 6)
    logprobs = reconstruct_13way_logprobs(logit_empty, logits_color, logits_type)
    assert logprobs.shape == (5, 13)
    # (a) softmax over the last dim sums to 1
    assert torch.allclose(torch.softmax(logprobs, dim=-1).sum(dim=-1), torch.ones(5), atol=1e-5)
    # since these are already normalised log-probabilities, exp() also sums to 1
    assert torch.allclose(logprobs.exp().sum(dim=-1), torch.ones(5), atol=1e-5)

def test_reconstruct_argmax_unambiguous():
    # (b) a very confident "non-empty, black, knight" should argmax to TARGET_MAP["n"].
    # sigmoid(logit_empty) == P(piece), so a confident piece has a large POSITIVE logit_empty.
    logit_empty = torch.tensor([10.0])                                      # P(piece) ~ 1
    logits_color = torch.tensor([[-10.0, 10.0]])                            # black
    logits_type = torch.tensor([[-10.0, -10.0, -10.0, -10.0, 10.0, -10.0]])  # N (type index 4)
    logprobs = reconstruct_13way_logprobs(logit_empty, logits_color, logits_type)
    assert logprobs.argmax(dim=-1).item() == TARGET_MAP["n"]

def test_reconstruct_occupancy_direction():
    # Guards the occupancy sign: sigmoid(logit_empty) == P(piece) (empty head trained on is_piece).
    color = torch.tensor([[10.0, -10.0]])                             # white
    king = torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0, -10.0]])  # K
    # confident piece (large +ve logit_empty) -> argmax is a piece, not empty
    piece = reconstruct_13way_logprobs(torch.tensor([10.0]), color, king)
    assert piece.argmax(dim=-1).item() == TARGET_MAP["K"]
    # confident empty (large -ve logit_empty) -> argmax is empty regardless of the color/type heads
    empty = reconstruct_13way_logprobs(torch.tensor([-10.0]), color, king)
    assert empty.argmax(dim=-1).item() == TARGET_MAP["empty"]

def test_reconstruct_matches_cross_entropy():
    # (c) feeding the output through CrossEntropyLoss reproduces -log(p_target) computed
    # directly from the reconstructed probabilities.
    torch.manual_seed(1)
    logit_empty = torch.randn(4)
    logits_color = torch.randn(4, 2)
    logits_type = torch.randn(4, 6)
    logprobs = reconstruct_13way_logprobs(logit_empty, logits_color, logits_type)
    target = torch.tensor([TARGET_MAP["empty"], TARGET_MAP["K"], TARGET_MAP["n"], TARGET_MAP["p"]])

    ce = nn.CrossEntropyLoss()(logprobs, target)
    probs = logprobs.exp()
    manual = -torch.log(probs[torch.arange(4), target]).mean()
    assert torch.allclose(ce, manual, atol=1e-5)
