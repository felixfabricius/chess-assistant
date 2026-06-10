from pathlib import Path

from chess_assistant.camera import capture_image
from chess_assistant.vision import infer_fen_from_image

def main() -> None:
    image_path = capture_image()
    print(f"Saved image to: {image_path}")

    # Test this with default 
    # board_position = infer_fen_from_image(image_path)
    board_position = infer_fen_from_image(Path("data") / "raw_images" / "test_image.jpg")
    print(f"Board position: {board_position}")

if __name__ == "__main__":
    main()