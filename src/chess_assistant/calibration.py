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

def position_robot(height, pitch):
    with ReachyMini(media_backend="default") as mini:
        pose = make_safe_pose(
            height,
            pitch
        )[0]

        mini.goto_target(pose, duration=MOVE_DURATION)

        return

def click_board_corners(frame) -> dict[str, list[int]]:
    """
    Let the user click the four semantic board corners.

    Click order:
    1. a1
    2. a8
    3. h8
    4. h1

    Returns:
        {
            "a1": [x, y],
            "a8": [x, y],
            "h8": [x, y],
            "h1": [x, y],
        }
    """
    corner_labels = ["a1", "a8", "h8", "h1"]
    corners: dict[str, list[int]] = {}

    display = frame.copy()
    window_name = "Click board corners: a1, a8, h8, h1"

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(corners) >= len(corner_labels):
            return

        label = corner_labels[len(corners)]
        corners[label] = [x, y]

        cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            display,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        print(f"Clicked {label}: ({x}, {y})")
        cv2.imshow(window_name, display)

    print("Click the board corners in this order:")
    print("1. a1")
    print("2. a8")
    print("3. h8")
    print("4. h1")
    print("Press ESC to cancel.")

    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, mouse_callback)

    while len(corners) < len(corner_labels):
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            raise RuntimeError("Corner clicking cancelled.")

    cv2.destroyWindow(window_name)
    return corners

def calibrate(setup_dir: Path = Path("data") / "raw_images"):
    height_mm = OPT_HEIGHT_MM
    pitch_deg = OPT_PITCH_MM

    last_sent_height = None
    last_sent_pitch = None

    corners_data_available = False

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
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

                    raw_image_path = setup_dir / "raw.png"
                    metadata_path = setup_dir / "calibration_metadata.json"

                    # Freeze current frame and save it
                    frozen_frame = frame.copy()
                    cv2.imwrite(str(raw_image_path), frozen_frame)

                    # Let user click semantic board corners
                    corners_px = click_board_corners(frozen_frame)

                    corners_data = {
                        "corner_order": ["a1", "a8", "h8", "h1"],
                        "corners_px": corners_px,
                        "warp_convention": {
                            "a8": [0, 0],
                            "h8": ["board_size_px", 0],
                            "h1": ["board_size_px", "board_size_px"],
                            "a1": [0, "board_size_px"],
                        },
                        "notes": (
                            "Warp convention assumes that a8 is in top left of image."
                        ),
                        "raw_image_path": str(raw_image_path)
                    }
                    print(f"Saved calibration image to: {raw_image_path}")
                    print(f"Saved calibration metadata to: {metadata_path}")
                    print(f"height_mm={height_mm}, pitch_deg={pitch_deg}")
                    print(f"corners_px={corners_px}")

                    corners_data_available = True
                    break
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

        return {
            "height": last_sent_height,
            "pitch": last_sent_pitch
        } | corners_data if corners_data_available else {}


if __name__ == "__main__":
    calibrate()
