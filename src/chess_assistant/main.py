from pathlib import Path

from chess_assistant.setup import setup
from chess_assistant.camera import capture_image
from chess_assistant.vision import infer_fen_from_image


import json
from datetime import datetime

def main() -> None:
    setup_dir, pixel_coordinates = setup()
    image_dir = capture_image(setup_dir)
    print(f"Saved image to: {image_dir / "image.png"}")

    """
    TODO: slice the image into squares. This requires: coordinates of the four corners.
    These can be stored in the setup_dir metadata (or perhaps also as local variables?)
    """

    # Test this with default 
    # board_position = infer_fen_from_image(image_path)
    model = "claude-opus-4-8"
    prompt_version = 0
    board_position = infer_fen_from_image(
        image_dir / "image.png",
        model=model,
        prompt_version=prompt_version
    )
    print(f"Board position: {board_position}")
    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "prompt_version": prompt_version,
        "raw_model_output": board_position
    }
    metadata_path = image_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()