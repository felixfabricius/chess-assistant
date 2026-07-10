import numpy as np
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

# This needs to detect events, so kind of needs to run continually
class AntennasInputDetector:
    def __init__(self, mini):
        self.mini = mini

        mini.goto_target(antennas=np.deg2rad([45, 45]), duration=0.5) # TODO: adjust these buttons

    def detect_input(self, antenna: str):
        assert antenna in ["left", "right"]
        _, antenna_positions = self.mini.get_current_joint_positions()
        # The first return value here is head_positions
        # Antenna positions = [rad, rad] - the actual measured angles

        # Needs to return None, "down", "up"
        return input

