import yaml
import os
from datetime import datetime
from pathlib import Path
import json

from chess_assistant.calibration import calibrate, position_robot

def setup():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    root_data_folder = Path(config.get("root_data_folder", "data"))

    if config.get("setup_folder", {}).get("create_new"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        setup_dir = root_data_folder / timestamp

        setup_dir.mkdir(parents=True)

        setup_data= (
            config.get("setup_folder", {}).get("measurements", {}) 
            | {key: config.get("setup_folder", {}).get(key) for key in ["chessboard", "documentation_image"]}
        )
        calibration_data = calibrate(setup_dir)
        
        with open(setup_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(setup_data | calibration_data, f, indent=2)

        pixel_coordinates = {field: calibration_data[field] for field in ["a1", "a8", "h8", "h1"]}
    
    else:
        # assert that root data folder exists and contains at least one file
        assert root_data_folder.is_dir()
        folders = os.listdir(root_data_folder)
        assert len(folders) > 0
        
        # This will return the folder associated with latest timestamp
        setup_dir = root_data_folder / sorted(folders)[-1]

        with open(setup_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Position robot correctly, and return the coordinates
        position_robot(metadata["height"], metadata["pitch"])

        pixel_coordinates = {field: metadata[field] for field in ["a1", "a8", "h8", "h1"]}

    return setup_dir, pixel_coordinates


