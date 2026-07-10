"""
This script is able to take a photo, send it to some LLM, and return a board position.
This should be testable. (And ideally I also store outputs of this, so I can later run
my own training.)
"""
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, make_dataclass
import anthropic
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

import numpy as np


from omegaconf import OmegaConf, DictConfig

from chess_assistant.config import SQUARES
from chess_assistant.model.config import INVERSE_TARGET_MAP, TARGET_MAP, reconstruct_13way_logprobs, TOP_LEFT_OHE_MAP
from chess_assistant.model.data import EVAL_TRANSFORM

load_dotenv()

PROMPTS = {
    0: (
        "You are looking at a physical chess board."
        "Return only the board position as a FEN board string, "
        "not the full FEN. Example format (if in starting position): "
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR. "
        "Do not include side to move, castling rights, move counters, or explanation."
        "CAREFULLY inspect each of the 64 squares individually to identify which piece - if any - "
        "is located there."
    ),
    1: (
        "You are classifying one square from a chessboard image.\n"
        "The target square is marked with a red corner markers. Other pieces and neighbouring squares may be visible because the crop includes padding.\n"
        "Classify only the chess piece whose BASE is on the highlighted target square. Ignore all other visible pieces.\n"
        "Return exactly one label from:\n"
        "empty, K, Q, R, B, N, P, k, q, r, b, n, p,\n"
        "where the letter corresponds to the piece in FEN notation, e.g. K stands for white king."
        "Return nothing but this label."
    )
}

@dataclass
class SquareEstimate:
    image_path: Path | None = None
    copied: bool = False
    copied_from: Path | None = None
    K: float = 0
    Q: float = 0
    R: float = 0
    B: float = 0
    N: float = 0
    P: float = 0
    k: float = 0
    q: float = 0
    r: float = 0
    b: float = 0
    n: float = 0
    p: float = 0
    empty: float = 0

BoardEstimate = make_dataclass(
    "BoardEstimate",
    [(square, SquareEstimate | None, None) for square in SQUARES]
)

def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def infer_media_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    if suffix not in types:
        raise ValueError(f"Unsupported image type: {suffix}")
    return types[suffix]

def infer_fen_from_image(image_path: Path, model: str = "claude-opus-4-8", prompt_version: int = 0) -> str:
    client = anthropic.Anthropic()

    prompt = PROMPTS[prompt_version]

    message = client.messages.create(
        #model="claude-sonnet-4-6",
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": infer_media_type(image_path),
                            "data": encode_image_base64(image_path),
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    )

    return message.content[0].text

class BoardEstimator:
    def __init__(self, model_type: str = "CNN", config: DictConfig | None = None, calibration_metadata_path: Path | None = None, model = None, device = None):
        """
        Keep track of:
        - recent board estimate
        - 

        To use:
        - iterate over each of the squares

        - look at fields of square in recent board estimate:
          - image path -> load image; could perhaps also store the array in memory right away? - though this could get quite large.
            then compare this to image for the current piece (accessible via squares_folder)
            copy estimates over if they match
            
          - 
        """
        assert model_type in ["CNN", "LLM"]
        self.board_estimate = BoardEstimate()
        if model_type == "LLM":
            assert config is not None
            self.model_version = config.vision.get("model_version", "claude-opus-4-8")
            self.prompt_version = config.vision.get("prompt_version", 1)
            self.client = anthropic.Anthropic()
        else:
            assert calibration_metadata_path is not None
            assert model is not None
            assert device in ["cpu", "cuda", None, torch.device("cpu"), torch.device("cuda")]
            self.device = torch.device(device) if device is not None else torch.device("cpu")
            with open(calibration_metadata_path, "r") as f:
                calibration_metadata = json.load(f)
            # The model's only metadata: which board corner is top-left in the image.
            self.top_left_corner = calibration_metadata["camera_natural_orientation"]["order"]["tl"]
            self.model = model.to(self.device)
        self.model_type = model_type

    def estimate_square(self, image_path: Path) -> SquareEstimate: 
        if self.model_type == "LLM":
            image_path = image_path.parent / (image_path.stem + "_annotated" + image_path.suffix)
            print(image_path) 
            message = self.client.messages.create(
                model=self.model_version,
                max_tokens=128,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": infer_media_type(image_path),
                                    "data": encode_image_base64(image_path),
                                }
                            },
                            {"type": "text", "text": PROMPTS[self.prompt_version]}
                        ]
                    }
                ]
            )
            breakpoint()
            square_estimate = SquareEstimate(
                image_path=image_path,
                copied=False,
                copied_from=None
            )
            print(message.content[0].text)
            try:
                setattr(square_estimate, message.content[0].text, 1.)
            except Exception as e:
                print(e)

            return square_estimate

        else:
            square = image_path.stem # TODO: this might be brittle if I change conventions;
            # and e.g. image_path is no longer .../{square}.png
            square_dir = image_path.parent
            setup_dir = square_dir.parent.parent.parent

            # Metadata: one-hot of which board corner is top-left in the image. Must match
            # training (model/data.py) - both use TOP_LEFT_OHE_MAP.
            metadata = torch.zeros(1, 4, dtype=torch.float32)
            metadata[0, TOP_LEFT_OHE_MAP[self.top_left_corner]] = 1
            metadata = metadata.to(self.device)
            assert metadata.shape == (1, 4)

            # For the image
            image = np.load(square_dir / f"{square}_masked.npy")
            rgb = image[..., :3]
            mask = tv_tensors.Mask(image[..., 3])
            rgb = EVAL_TRANSFORM(rgb)
            mask = EVAL_TRANSFORM(mask).unsqueeze(dim=0)
            image = torch.cat([rgb, mask]).unsqueeze(dim=0).to(self.device)
            assert image.shape == (1, 4, 144, 144)
            
            # Now turn the model output into the square prediction;
            # in both cases we store logit-like values (raw logits or reconstructed
            # log-probabilities), which game.py feeds into its own CrossEntropyLoss.
            square_estimate = SquareEstimate(
                image_path=image_path,
                copied=False,
                copied_from=None
            )

            if hasattr(self.model, "empty_head"):
                # Multi-head model (model 3): recombine the three heads into 13-way
                # log-probabilities. softmax(log p) == p and the reconstructed probs sum to
                # 1, so log p is a drop-in for the old single-head logits under the softmax
                # game.py re-applies downstream.
                logit_empty, logits_color, logits_type = self.model(image, metadata)
                logprobs = reconstruct_13way_logprobs(
                    logit_empty.squeeze(0), logits_color.squeeze(0), logits_type.squeeze(0)
                )
                for label, idx in TARGET_MAP.items():
                    setattr(square_estimate, label, logprobs[idx].item())
            else:
                logits = self.model(image, metadata).squeeze()
                # Shape of the non-squeezed logits would be (1, 13)
                for label in INVERSE_TARGET_MAP.keys():
                    # label takes values in 0, ..., 12
                    setattr(square_estimate, INVERSE_TARGET_MAP[label], logits[label].item())

            return square_estimate

    def estimate_board(self, squares_dir):
        """
        Declar new BoardEstimate object.
        For each square:
            - check if image has changed relative to the last image (see recent_board.square.image_path) to compare
            - if no:
                - then create a copy.deepcopy of estimates for that square & modify the copied_from 
                  (Maybe we don't even need a copy.deepcopy because it's okay if we overwrite?)
                  I think that's quite plausiblel actually
            - if yes:
                - classify the individual square. That shold be a separate method.
        """
        for square in SQUARES:
            image_path = squares_dir / square / f"{square}.png"
            if (
                getattr(self.board_estimate, square) # these are initialised as None
                and 1 == 2 # TODO # check similarity between images:
            ):            
                # Copy over
                # Set the copied_from attribute
                # Note that this also overwrites self.recent_board. 
                # So maybe no need to pretend we have two separate objects here?
                # Just use self.board for everything?
                """
                TODO: fix this section. Not sure this is correct. (Also perhaps think about valid 
                condition in the if-statement.)
                When fixing, carefully note that getattr and setattr apparently do not support
                nested attribute access.

                To search for such patterns: getattr\([^)\n]*\.
                """
                square_estimate = getattr(self.recent_board, square)
                setattr(
                    self.board_estimate, f"{square}.copied_from",
                    (
                        getattr(self.recent_board, f"{square}.copied_from")
                        if getattr(self.recent_board, f"{square}.copied")
                        else getattr(self.recent_board, f"{square}.image_path")
                    )
                )
                setattr(self.board_estimate, f"{square}.image_path", image_path)
                setattr(self.board_estimate, f"{square}.copied", True)
            else:
                # TODO: perhaps also pass additional info?
                # Like pixel position of square, and some metadata about the robot position?
                # (I think this is particularly relevant for our own model, which might be able to learn )
                # valuable things from this.
                square_estimate = self.estimate_square(image_path)
                setattr(self.board_estimate, square, square_estimate)

        return self.board_estimate


if __name__ == "__main__":
    """
    Pass:
    1. squares_dir; example: data/raw_images/squares
    2. config_path: config.yaml
    3. squares separated by spaces, e.g. a1 a2 h8

    Example:
    uv run python -m chess_assistant.vision data/raw_images/squares config.yaml a1 d4 d5 a8 h8
    """
    import sys
    assert len(sys.argv) > 2
    
    squares_dir = Path(sys.argv[1])

    config = OmegaConf.load(sys.argv[2])

    SQUARES = [square for square in sys.argv[3:]]
    print(SQUARES)
    BoardEstimate = make_dataclass(
        "BoardEstimate",
        [(square, SquareEstimate | None, None) for square in SQUARES]
    )
    board_estimator = BoardEstimator(config)
    print(board_estimator)
    
    board_estimate = board_estimator.classify_board(squares_dir)
    print(board_estimate)
