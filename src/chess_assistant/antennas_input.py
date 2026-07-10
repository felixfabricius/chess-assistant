import json
import numpy as np

from pathlib import Path


"""
To implement here:
- for each of the two antennas, need to specify degree amount that makes them stand out horizontally. 
  (Note that head can be tilted which can make this awkward in principle.)
- also for each of the antennas, need ways to detect: "antenna is used as a button"
  i.e. some radian amount that if we exceed or fall below, we know: this antenna has been pressed up / down
- finally, need to wire this into main by constantly monitoring for events. Perhaps this can be done using 
  a while loop and perhaps time.sleep()


"""

SIDE_MAP = {
    "right": 0,
    "left": 1
}

INVERSE_SIDE_MAP = {0: "right", 1: "left"}

THRESHOLD_DEGREES = 3

# This needs to detect events, so kind of needs to run continually
class AntennasInputDetector:
    def __init__(self, mini, calibration_metadata_path, baseline_rotation=85):
        self.mini = mini

        # Assume that robot sits NEXT to the board centrally, allowing
        # both players to easily operate the antennas as buttons.
        # Then the top-left square from the robots perspective should be either
        # h8, in which case white is on its right side and black on its left side
        # or a1, in which case the converse is true.
        # The robot can also be set up at other positions, but in that case
        # the "press antenna to submit move" interface is awkward and should 
        # be modified. 
        with open(calibration_metadata_path, "r") as f:
            calibration_metadata = json.load(f)
            tl = calibration_metadata["camera_natural_orientation"]["order"]["tl"]
            assert tl in ["a1", "h8"]
            if tl == "a1":
                self.side_to_play = "left"
            else:
                self.side_to_play = "right"

        # From the robot's perspective, the first argument here controls the right antenna
        # and the second one controls the left antenna.
        # And 0 degrees means antenna points straight upwards. Slightly >0 means anticlockwise
        # rotation. 
        self.baseline_rotation = 85
        mini.goto_target(antennas=np.deg2rad([-self.baseline_rotation, self.baseline_rotation]), duration=0.5)

    def detect_input(self, type: str) -> bool:
        breakpoint()
        # Look for one of two input patterns:
        # Move antenna down to make move; move antenna up to reject move
        assert type in ["move_made", "move_estimate_rejected"]

        _, antenna_positions = self.mini.get_current_joint_positions()
        
        antenna_position = antenna_positions[SIDE_MAP[self.side_to_play]]
        antenna_position = np.rad2deg(antenna_position)

        if type == "move_made":
            # If side to play is left, then antenna being pushed down corresponds to 
            # small anti-clockwise rotation, which is encoded as degrees slightly increasing.
            if self.side_to_play == "left":
                return antenna_position - self.baseline_rotation > THRESHOLD_DEGREES
            return -self.baseline_rotation - antenna_position > THRESHOLD_DEGREES
                # if side_to_play is right, then move_made requires that
                # antenna_position is more negative than -self.baseline_rotation
        
        if self.side_to_play == "left":
            return self.baseline_rotation - antenna_position > THRESHOLD_DEGREES
        return antenna_position - -self.baseline_rotation > THRESHOLD_DEGREES

    def reset_positions(self):
        self.mini.goto_target(antennas=np.deg2rad([-self.baseline_rotation, self.baseline_rotation]), duration=0.5)

    def switch_turn(self):
        self.side_to_play = INVERSE_SIDE_MAP[1 - SIDE_MAP[self.side_to_play]]