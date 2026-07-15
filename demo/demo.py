"""
TODO:
- Get actual images and moves; maybe two - three actually; (will it still be possible to do this by image? yes. Nested dictionary could be neat.)
- Also pass the correct previous_fens into the image.
- Perhaps get probabilities that move is correct?
  How to?
  Implcitly: Turn each Square Estimate into prob vector; Then
  calculate prob that all 64 squares have necessary value across all the values.
  Then normalise by sum of this over all 64 squares. 
  (Perhaps also return the unnormalised version? 
  Could actually be really interesting; and shows value of the decoding approach.)
- Evaluation: (extended)
  - look at board estimate in particular and get:
    fraction of squares that are correct; perhaps print a big table with all the squares, predicted
    and whether correct / incorrect
  - and also probabilities of the different moves?
"""
from pathlib import Path
import cv2
import json
from safetensors import load_file

from chess_assistant.image_processing import Processor
from chess_assistant.vision import BoardEstimate, BoardEstimator
from chess_assistant.game import ChessGame

def load_image(img_path: Path):
    # This should be just a png of the whole board
    # The calibration should just affect the setup-metadata. That should be optional.
    return cv2.imread(img_path) 

def load_metadata(metadata_path: Path) -> dict:
    # Metadata should contain keys:
        # "camera_intrinsics"
        # metadata["camera_natural_orientation"]["order"]
        # metadata["actual_corners_px"]
        # metadata["extended_corners_px"]
        # "extended_center_ox"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata

def warp_image(img_path: Path, calibration_metadata_path: Path) -> tuple[Processor, Path]:
    img_processor = Processor(calibration_metadata_path)
    warped_image_path = img_processor.warp(img_path, "out/warped_image.png")
    return img_processor, warped_image_path


def cutout_squares(img_processor: Processor, warped_image_path: Path) -> Path:
    squares_dir = img_processor.cutout(warped_image_path)
    return squares_dir


def estimate_move(
    squares_dir: Path,
    calibration_metadata_path: Path,    
    # FEN encodes a chess position. To detect a move, we look at 
    # all the moves that are legal at a given chess position, and then
    # use our "board_estimate" to determine which of them is most likely.
    previous_fen: str 
) -> tuple[list[str], BoardEstimate]:
    board_estimator = BoardEstimator(
        model_type="CNN",
        calibration_metadata_path=calibration_metadata_path,
        model_weights_path=Path("weights/model_state_dict.safetensors")
    )
    board_estimate = board_estimator.estimate_board(squares_dir)
    
    game = ChessGame(model_type="CNN", fen=previous_fen)
    # This is a list of all possible moves, ordered descendibgly by
    # how likely they are to be the one correct move given the previous
    # board position and an image of the current board state.
    estimated_moves = game.estimate_move(board_estimate)
    return list(estimated_moves.keys()), board_estimate


def evaluate_estimate()

# %%
IMG_PATH = Path("demo/assets/image/image.png")
CALIBRATION_METADATA_PATH = Path("demo/assets/image/metadata.json")
PREVIOUS_FEN = "PLACEHOLDER"

# %%
img_processor, warped_image_path = warp_image(IMG_PATH, CALIBRATION_METADATA_PATH)
print((
    "The image has been warped based on the labelled board corners "
    "('actual_corners_px' and 'extended_board_corners' in the calibration metadata). "
    f"\nThe warped image is at: {IMG_PATH}"
))

#%%
squares_dir = cutout_squares(img_processor, warped_image_path)
print((
    "Individual squares have been cut out and saved in the directory: "
    f"{squares_dir}.\n"
    "The position of the actual chess board square within its "
    "cutout tends to differ across squares: to fully capture pieces standing "
    "on squares, need to extend the cutout into a direction which depends on the "
    "camera orientation and differs across squares."
))

#%%
estimated_moves, board_estimate = estimate_move(
    squares_dir,
    CALIBRATION_METADATA_PATH
    PREVIOUS_FEN
)
print(
    "For the 64 squares the model estimated how which piece (if any) is located there. \n",
    "Based on those estimates, the legal moves from the previous position were ordered by ",
    "how likely they are to have lead to the board position seen in the image. \n",
    "The most likely moves are: \n",
    *tuple([f'{i + 1}: {move}' for i, move in enumerate(estimated_moves[:3])]),
    sep=""
)
