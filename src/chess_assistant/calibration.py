from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import cv2
import time
from datetime import datetime
from pathlib import Path

MIN_HEIGHT_MM = -40
MAX_HEIGHT_MM = 21

MIN_PITCH_DEG = -28
MAX_PITCH_DEG = 28

HEIGHT_STEP_MM = 2
PITCH_STEP_DEG = 2

OPT_HEIGHT_MM = 8
OPT_PITCH_MM = 26

MOVE_DURATION = 0.25

def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))

def make_safe_pose(height_mm: float, pitch_deg: float):
    height_mm = clamp(height_mm, MIN_HEIGHT_MM, MAX_HEIGHT_MM)
    pitch_deg = clamp(pitch_deg, MIN_PITCH_DEG, MAX_PITCH_DEG)

    pose = create_head_pose(
        z=height_mm,
        pitch=pitch_deg,
        mm=True,
        degrees=True,
    )

    return pose, height_mm, pitch_deg

def position_robot():
    with ReachyMini(media_backend="default") as mini:
        pose = make_safe_pose(
            OPT_HEIGHT_MM,
            OPT_PITCH_MM
        )[0]

        mini.goto_target(pose, duration=MOVE_DURATION)

        return

def main(output_dir: Path = Path("data") / "raw_images"):
    height_mm = OPT_HEIGHT_MM
    pitch_deg = OPT_PITCH_MM

    last_sent_height = None
    last_sent_pitch = None

    with ReachyMini(media_backend="default") as mini:
        while True:
            frame = mini.media.get_frame()

            if frame is not None:
                cv2.imshow("Reachy board view", frame)

            key = cv2.waitKey(1) & 0xFF

            new_height_mm = height_mm
            new_pitch_deg = pitch_deg

            if key == ord("w"):
                new_height_mm += HEIGHT_STEP_MM
            elif key == ord("s"):
                new_height_mm -= HEIGHT_STEP_MM
            elif key == ord("i"):
                new_pitch_deg += PITCH_STEP_DEG
            elif key == ord("k"):
                new_pitch_deg -= PITCH_STEP_DEG
            elif key == ord(" "):
                if frame is not None:
                    timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
                    output_path = output_dir / f"reachy_board_{timestamp}.png"
                    cv2.imwrite("calibration_view.jpg", frame)
                print(f"Saved. height_mm={height_mm}, pitch_deg={pitch_deg}")
            elif key == ord("q"):
                break
            else:
                continue

            # Clamp requested values before sending to robot
            pose, safe_height, safe_pitch = make_safe_pose(
                new_height_mm,
                new_pitch_deg,
            )

            # If clamping changed the requested value, tell yourself
            if safe_height != new_height_mm or safe_pitch != new_pitch_deg:
                print(
                    "Requested pose outside safe range. "
                    f"Clamped to height={safe_height}, pitch={safe_pitch}"
                )

            # Avoid spamming the robot with the same command repeatedly
            if (
                safe_height != last_sent_height
                or safe_pitch != last_sent_pitch
            ):
                print(f"Moving to height={safe_height}, pitch={safe_pitch}")

                try:
                    mini.set_target(
                        head=pose,
                        #duration=MOVE_DURATION,
                        body_yaw=None,
                    )

                    height_mm = safe_height
                    pitch_deg = safe_pitch
                    last_sent_height = safe_height
                    last_sent_pitch = safe_pitch

                    time.sleep(MOVE_DURATION)

                except Exception as e:
                    print("Move failed:", e)
                    print(
                        f"Keeping previous pose: "
                        f"height={height_mm}, pitch={pitch_deg}"
                    )

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
