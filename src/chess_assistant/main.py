from pathlib import Path
from omegaconf import OmegaConf

from chess_assistant.setup import setup
from chess_assistant.camera import capture_image
from chess_assistant.vision import BoardEstimator
from chess_assistant.game import ChessGame
from chess_assistant.image_processing import Processor


import json
from datetime import datetime

def main() -> None:
    config = OmegaConf.load("config.yaml")

    setup_dir, pixel_coordinates = setup()
    image_processor = Processor(setup_dir / "calibration_metadata.json", "config.yaml")
    board_estimator = BoardEstimator(config)
    
    game = ChessGame()

    # Build game loop
    i = 0
    while True:
        # Capture image from camera
        image_dir = capture_image(setup_dir)
        # Warp and cut out squares
        warped_image_path = image_processor.warp(image_dir / "image.png")
        squares_dir = image_processor.cutout(warped_image_path)
        board_estimation = BoardEstimator.classify_board(squares_dir)

        # Recognise chess board position
        (image_dir, config)
        
        # Classify individual squares
            # provide the squares folder.
            # for each square:
                # call model; generate and store output in some json file
        
        # Infer board position
        # Use all the predictions of the individual squares, to build board position
        if i > 5:
            break
        i += 1

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
    import sys
    if len(sys.argv) > 1:

    main()