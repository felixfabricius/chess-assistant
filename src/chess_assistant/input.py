import json
import numpy as np
import sys
import time

if sys.platform == "win32":
    import msvcrt
else:
    import termios
    import tty
    import select

from pathlib import Path
from reachy_mini import ReachyMini


"""
To implement here:
- for each of the two antennas, need to specify degree amount that makes them stand out horizontally. 
  (Note that head can be tilted which can make this awkward in principle.)
- also for each of the antennas, need ways to detect: "antenna is used as a button"
  i.e. some radian amount that if we exceed or fall below, we know: this antenna has been pressed up / down
- finally, need to wire this into main by constantly monitoring for events. Perhaps this can be done using 
  a while loop and perhaps time.sleep()
"""

# One input manager class
# Should support a few things:
    # Input via antennas (up / down separately)
    # Or input via keys (any key, a specific key)
# And also:
    # "See if there was any input in a given period of time"
    # Or: wait until there was input

SIDE_MAP = {
    "right": 0,
    "left": 1
}

INVERSE_SIDE_MAP = {0: "right", 1: "left"}

THRESHOLD_DEGREES = 3

# This needs to detect events, so kind of needs to run continually
class InputDetector:
    def __init__(
            self, 
            input_type: str = "keyboard", 
            target_key: str | None = " ",
            mini: ReachyMini | None = None,
            calibration_metadata_path: Path | None = None,
            baseline_rotation: int | None = 85,
            max_time: float = 1 # max time in hours to wait for 'necessary' events; note int is subtype of float
        ):
        assert input_type in ["keyboard", "robot"]
        self.input_type = input_type
        self.max_time = max_time
        if input_type == "keyboard":
            self.target_key = target_key
            self.platform = sys.platform

        else:
            assert isinstance(mini, ReachyMini)
            assert isinstance(calibration_metadata_path, Path)
            assert isinstance(baseline_rotation, int)

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
            self.baseline_rotation = baseline_rotation
            mini.goto_target(antennas=np.deg2rad([-self.baseline_rotation, self.baseline_rotation]), duration=0.5)

    def detect_input(self, type: str, alloted_time: float = 3, antenna_direction: str = "downwards") -> bool:
        # Type: necessary or optional
        # For necessary input: wait for input until it is received. Only then return from function.
        # For optional input: 
        # Only wait for a specific amount of time and return whether that input was received or not
        # from function once time has run out.

        # Look for one of two input patterns:
        # Move antenna down to make move; move antenna up to reject move
        assert type in ["necessary", "optional", "move_made", "move_estimate_rejected"]
        # "move_made" and "move_estimate_rejected are shortcuts" which map to
            # Nescessary input, antenna_movement = "down"
            # And opional input, antenna_movement = "up" respectively
        
        if type == "move_made":
            type = "necessary"
            antenna_direction = "downwards"
        elif type == "move_estimate_rejected":
            type = "optional"
            antenna_direction = "upwards"

        move_made = False
        alloted_time = self.max_time * 3600 if type == "necessary" else alloted_time # max time is in hours

        end_time = time.time() + alloted_time

        # Drop any keystrokes buffered during a previous phase so a stale press can't be
        # misread as fresh input (e.g. auto-rejecting the first suggested move).
        if self.input_type == "keyboard" and self.platform == "win32":
            while msvcrt.kbhit():
                msvcrt.getwch()

        while time.time() < end_time and not move_made:
            if self.input_type == "robot":
                assert antenna_direction in ["downwards", "upwards"]
                _, antenna_positions = self.mini.get_current_joint_positions()
            
                antenna_position = antenna_positions[SIDE_MAP[self.side_to_play]]
                antenna_position = np.rad2deg(antenna_position)

                if antenna_direction == "downwards":
                    # If side to play is left, then antenna being pushed down corresponds to 
                    # small anti-clockwise rotation, which is encoded as degrees slightly increasing.
                    if self.side_to_play == "left":
                        move_made = antenna_position - self.baseline_rotation > THRESHOLD_DEGREES
                    else:
                        move_made = -self.baseline_rotation - antenna_position > THRESHOLD_DEGREES
                        # if side_to_play is right, then move_made requires that
                        # antenna_position is more negative than -self.baseline_rotation
                else:
                    if self.side_to_play == "left":
                        move_made = self.baseline_rotation - antenna_position > THRESHOLD_DEGREES
                    else:
                        move_made = antenna_position - -self.baseline_rotation > THRESHOLD_DEGREES
        
            else: # input_type is 'keyboard'
                if self.platform == "win32":
                    if msvcrt.kbhit():
                        key = msvcrt.getwch()
                        if self.target_key is None or key == self.target_key:
                            move_made = True
                else:
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setcbreak(fd)
                        remaining = end_time - time.time()
                        rlist, _, _ = select.select([sys.stdin], [], [], remaining)
                        if rlist:
                            key = sys.stdin.read(1)
                            if self.target_key is None or key == self.target_key:
                                move_made = True
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            time.sleep(0.02)  # avoid a tight busy-loop while polling for input
        return move_made
        

    def reset_positions(self):
        if self.input_type == "robot":
            self.mini.goto_target(antennas=np.deg2rad([-self.baseline_rotation, self.baseline_rotation]), duration=0.5)

    def switch_turn(self):
        if self.input_type == "robot":
            self.side_to_play = INVERSE_SIDE_MAP[1 - SIDE_MAP[self.side_to_play]]


