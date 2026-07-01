import json
import polars as pl
from pathlib import Path
from datetime import datetime

from chess_assistant.calibration import calibrate

# Files we need

# hash table which maps from set-up to set-up directory
    # note that the calibration metadata should still be kept track off at the setup level
# csv file which keeps track of everything
def hash_setup(setup: dict[str, float]) -> str:
    return ""

def generate_data(
    setups_path: Path = "data/setups.json",
    csv_path: Path = "data/data.csv"
):
    with open(setups_path, "r", encoding="utf-8") as f:
        setups = json.load(f)

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


        # 


if __name__ == "__main__":
    generate_data()
