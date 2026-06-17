"""
Script which is able to connect to Reachys's camera and take photos.
"""
from pathlib import Path
from datetime import datetime

import cv2
from reachy_mini import ReachyMini

def capture_image(output_dir: Path) -> Path:
    """
    Capture on frame from Reachy's camera and save it as a PNG.
    """
    assert isinstance(output_dir, Path)
    assert output_dir.is_dir()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    image_dir = output_dir / f"reachy_board_{timestamp}"
    image_dir.mkdir()
    image_path = image_dir / "image.png"

    with ReachyMini(media_backend="default") as mini:
        frame = mini.media.get_frame()

    cv2.imwrite(str(image_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return image_dir
