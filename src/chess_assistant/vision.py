"""
This script is able to take a photo, send it to some LLM, and return a board position.
This should be testable. (And ideally I also store outputs of this, so I can later run
my own training.)
"""
import base64
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def infer_media_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    if suffix not in types:
        raise ValueError(f"Unsupported image type: {suffix}")
    return types[suffix]

def infer_fen_from_image(image_path: Path) -> str:
    client = anthropic.Anthropic()

    prompt = (
        "You are looking at a physical chess board."
        "Return only the board position as a FEN board string, "
        "not the full FEN. Example format (if in starting position): "
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR. "
        "Do not include side to move, castling rights, move counters, or explanation."
        "CAREFULLY inspect each of the 64 squares individually to identify which piece - if any - "
        "is located there."
    )

    message = client.messages.create(
        #model="claude-sonnet-4-6",
        model="claude-opus-4-8",
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