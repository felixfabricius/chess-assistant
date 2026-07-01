import json
import yaml
import polars as pl
from pathlib import Path
from datetime import datetime

from chess_assistant.calibration import calibrate
from chess_assistant.camera import capture_image
from chess_assistant.image_processing import Processor
from chess_assistant.config import SQUARES, PIECES

# Files we need

# hash table which maps from set-up to set-up directory
    # note that the calibration metadata should still be kept track off at the setup level
# csv file which keeps track of everything
def hash_setup(setup: dict[str, float]) -> str:
    return ""

def generate_data(
    setups_path: Path = "data/setups.json",
    csv_path: Path = "data/data.csv",
    config_path: Path = "config.yaml"
):
    with open(setups_path, "r", encoding="utf-8") as f:
        setups = json.load(f)

    #with open(config_path, "r", encoding="utf-8") as f:
    #    config = yaml.safe_load(f)

    data = pl.read_csv(csv_path)

    """
    Training generation loop:
    - input set-up data, submit set-up data (perhaps using enter)
    - hash the set-up data and find the correct set-up folder; if it doesn't yet exist, then create set-up folder
      and store calibration metadata:
        - as specified in calibrate: actual_corners px, extended_corners_px, a
          and stuff that identifies robot orientation; dist, height_base, orientation, offset, height_head, pitch
          ()
          perhaps more simply, we can just take: pixel positions of the 4 corners, and then OHE vector for which corner is in
          top left; reason: that should contain the exact same information, but is much easier to capture
          
          also: means that I don't need to manually measure orientation etc. 

          question then: how to efficiently switch between setups for the same board position? very difficult, but maybe that's fine

          just say: new setup and then calibrate again; then do a bunch of positions.

        - Perhaps for now also store: previous FEN, and current FEN. This is only possible for 
          cases where I am constrained to legal moves (and have been constrained for legal moves for entirety of current game)
          Store this as a boolean variable in "data"

          
          
    """

    while True:
        # Calibrate the robot; -> create new setup folder etc. 
        # Then show interface with chessboard, where I can freely move around pieces;
          # as well as some button (or simply key) for quit/save; and some button/key for new setup 
          # also need a button for capture. if capture, then want to take image, warp, preprocess etc.
          # and also manage labels

        # New setup -> calibrate
        setup_dir = Path("data") / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        calibration_data = calibrate(setup_dir)
        if calibration_data:
            calibration_metadata_path = setup_dir / "calibration_metadata.json"
        else:
            # TODO; in this case, something went wrong. Perhaps need to try again.
            raise NotImplementedError

        valid_position = True # set to false as soon as we allow non-valid position even once
        # for a given set-up

        # For a given set-up, initialise processor
        # When initialising, using the pixel coordinates in the calibration metadata, 
        # this infers how to warp the chessboard image, and how to pad. 
        # from then on, can use its warp and cutout methods to obtain individual square cutouts of warped image
        
        # Needs to be reinitialised with new calibration_metadata if setup changes
        image_processor = Processor(calibration_metadata_path, config_path)

        # When capturing an image:
        # This saves all the square cutouts
        image_dir = capture_image(setup_dir)

        # Add metadata here:
        image_metadata = {
            "valid_position": valid_position,
            "board_fen": None,# fen if valid position, else only the part that is linked to position; TODO
            "previous_fen": None # previous fen if valid position
        }

        warped_image_path = image_processor.warp(image_dir / "image.png")
        squares_dir = image_processor.cutout(warped_image_path)

        # How does this need to be modified? / what needs to be added: labels

        # Only capture an image when we've updated the chess-position
        # is it sufficient to store the label in the master csv, or also add it to each
        # individual squares directory?
            # maybe for robustness add to each individual squares directory, but for simplicity
            # let's not do that right now
        
        # For each image, create a bunch of new rows and add them to our dataframe

        # Need 64 new rows for each square; and then need to link to the square cutout
        # need to access the label! need a function which uses our chessboard representation to 
        # access figure that is on a given square 
        # need to 

        labels = [piece_at(square) for square in SQUARES] # TODO; piece_at
        images = [str(squares_dir / square / f"{square}.png") for square in SQUARES]
         # only if the "enable invalid position thing "


        new_rows = pl.DataFrame(
            "square": SQUARES,
            "label": labels,
            "image": images,
            "calibration_metadata_path": calibration_metadata_path,
            "valid_position": valid_position,
            ""
        )


if __name__ == "__main__":
    generate_data()
