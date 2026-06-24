"""
This script is able to take a photo, send it to some LLM, and return a board position.
This should be testable. (And ideally I also store outputs of this, so I can later run
my own training.)
"""
import base64
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, make_dataclass
import anthropic

from omegaconf import OmegaConf, DictConfig

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
        "The target square is marked with a red outline/corner markers. Other pieces and neighbouring squares may be visible because the crop includes padding.\n"
        "Classify only the chess piece whose base is on the highlighted target square. Ignore all other visible pieces.\n"
        "Return exactly one label from:\n"
        "empty, white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king, "
        "black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king.\n"
        "Also return a confidence between 0 and 1."
    )
}

FILES = ["a", "b", "c", "d", "e", "f", "g", "g"]
RANKS = [str(i) for i in range(1, 9)]
SQUARES = [file + rank for file in FILES for rank in RANKS]

@dataclass
class SquarePrediction:
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
    empty: float = 1

BoardPrediction = make_dataclass(
    "BoardPrediction",
    [(square, SquarePrediction, SquarePrediction()) for square in SQUARES]
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

def classify_squares(squares_path: Path, config: DictConfig):
    # Maintain latest folder for which we have predictions; does this warrant turning this into a class?
    # Also:
    # This would allows us to not have to read things out repeatedly, but rather maintain one data structure which keeps track of
    # all the predictions.

    # What should that data structure look like:
    # 

    #if config.vision.get("model", "LLM") == "LLM":
        

    else:

class ChessPositionClassifier:
    def __init__(self, squares_dir, config):
        """
        Keep track of:
        - recent board prediction
        - 

        To use:
        - iterate over each of the squares

        - look at fields of square in recent board prediction:
          - image path -> load image; could perhaps also store the array in memory right away? - though this could get quite large.
            then compare this to image for the current piece (accessible via squares_folder)
            copy predictions over if they match
            
          - 
        """
        self.squares_dir = squares_dir
        self.recent_board: BoardPrediction = BoardPrediction()
        self.model = config.vision.get("model", "LLM")
    
    def classify_square(self, image_path):
        return

    def classify_board(self, squares_dir):
        """
        Declar new BoardPrediction object.
        For each square:
            - check if image has changed relative to the last image (see recent_board.square.image_path) to compare
            - if no:
                - then create a copy.deepcopy of predictions for that square & modify the copied_from 
                  (Maybe we don't even need a copy.deepcopy because it's okay if we overwrite?)
                  I think that's quite plausiblel actually
            - if yes:
                - classify the individual square. That shold be a separate method.
        """
        board_prediction = self.recent_board

        for square in SQUARES:
            image_path = squares_dir / square / f"{square}.png"
            if getattr(self.recent_board, f"{square}.image_path") and 1 == 2: # TODO # check similarity between images:
                # Copy over
                # Set the copied_from attribute
                # Note that this also overwrites self.recent_board. 
                # So maybe no need to pretend we have two separate objects here?
                # Just use self.board for everything?
                setattr(
                    board_prediction, f"{square}.copied_from",
                    (
                        getattr(self.recent_board, f"{square}.copied_from")
                        if getattr(self.recent_board, f"{square}.copied")
                        else getattr(self.recent_board, f"{square}.image_path")
                    )
                )
                setattr(board_prediction, f"{square}.image_path", image_path)
                setattr(board_prediction, f"{square}.copied", True)
            else:
                # TODO: perhaps also pass additional info?
                # Like pixel position of square, and some metadata about the robot position?
                # (I think this is particularly relevant for our own model, which might be able to learn )
                # valuable things from this.
                setattr(self.board_prediction, square, self.classify_square(image_path))

                
                setattr(board_prediction, f"")
                raise NotImplementedError


            # Classify individual square


        return

    def extract_position(self):
        return

