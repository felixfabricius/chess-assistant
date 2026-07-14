"""How the players talk back to the robot: either the antennas or the keyboard.

:class:`InputDetector` exposes one primitive -- "was there an input?" -- in two flavours:
*necessary* input, which blocks until it arrives, and *optional* input, which gives up after a
short window. That is enough for the whole game loop: waiting for a player to finish their move
is necessary input, and catching a rejection of a suggested move is optional input.

With ``input_type="robot"`` the two antennas are the buttons: the player to move pushes their
antenna down to confirm, or up to reject the robot's suggestion. With ``input_type="keyboard"``
any key (or one specific key) stands in for that, which is beneficial for camera angle stability.
(Pushing antennas tends to result in the robot head returning to a marginally different position,
which disturbs image processing.)
"""

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


SIDE_MAP = {
    "right": 0,
    "left": 1
}

INVERSE_SIDE_MAP = {0: "right", 1: "left"}

# How far an antenna has to travel off its baseline before we call it a deliberate press. Small
# enough to be an easy flick, large enough not to trigger on the slop in a resting antenna.
THRESHOLD_DEGREES = 3


class InputDetector:
    """Polls whichever input source is configured until it sees a press (or times out)."""

    def __init__(
            self, 
            input_type: str = "keyboard", 
            target_key: str | None = " ",
            mini: ReachyMini | None = None,
            calibration_metadata_path: Path | None = None,
            baseline_rotation: int | None = 85,
            max_time: float = 1 # max time in HOURS to wait for a 'necessary' event
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

            # Assume that the robot sits NEXT to the board centrally, allowing
            # both players to easily operate the antennas as buttons.
            # Then the top-left square from the robot's perspective should be either
            # h8, in which case white is on its right side and black on its left side,
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
            # And 0 degrees means the antenna points straight upwards. Slightly >0 means
            # anticlockwise rotation.
            # The baseline (85 degrees) therefore splays both antennas out roughly horizontally,
            # which leaves them room to travel in either direction from a visible resting pose.
            self.baseline_rotation = baseline_rotation
            mini.goto_target(antennas=np.deg2rad([-self.baseline_rotation, self.baseline_rotation]), duration=0.5)

    def detect_input(self, type: str, alloted_time: float = 3, antenna_direction: str = "downwards") -> bool:
        """Wait for an input and report whether one arrived.

        ``type`` is one of:

        - ``"necessary"`` -- block until the input arrives (giving up only after ``max_time``
          hours). Used when the game cannot proceed without the players, e.g. waiting for them
          to finish their move on the physical board.
        - ``"optional"`` -- wait ``alloted_time`` seconds and report whether the input arrived.
          Used for the review window, where silence means "no objection".
        - ``"move_made"`` / ``"move_estimate_rejected"`` -- shortcuts for the two combinations
          the game loop actually uses: necessary + antenna down, and optional + antenna up.

        The two antenna gestures are deliberately opposite: push the antenna DOWN to confirm a
        move, and UP to reject the robot's suggestion.
        """
        assert type in ["necessary", "optional", "move_made", "move_estimate_rejected"]

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
                    # The two antennas rest at opposite baselines (+85 and -85), so "down" is a
                    # different sign for each. For the left antenna it is a small anti-clockwise
                    # rotation, i.e. degrees increasing past +baseline; for the right one it is
                    # the mirror image, i.e. degrees dropping below -baseline. Hence the two
                    # comparisons look different but mean the same gesture.
                    if self.side_to_play == "left":
                        move_made = antenna_position - self.baseline_rotation > THRESHOLD_DEGREES
                    else:
                        move_made = -self.baseline_rotation - antenna_position > THRESHOLD_DEGREES
                else:
                    # Same idea for "up", with both comparisons flipped.
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
        """Splay both antennas back out to the baseline, ready for the next press."""
        if self.input_type == "robot":
            self.mini.goto_target(antennas=np.deg2rad([-self.baseline_rotation, self.baseline_rotation]), duration=0.5)

    def switch_turn(self):
        """Hand the buttons to the other player, so we watch their antenna next.

        A no-op in keyboard mode, where both players share the same keyboard.
        """
        if self.input_type == "robot":
            self.side_to_play = INVERSE_SIDE_MAP[1 - SIDE_MAP[self.side_to_play]]


