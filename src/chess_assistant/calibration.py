import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# reachy_mini (the `robot` dependency group) is imported lazily inside the functions that need
# the live robot, so the pure calibration helpers below stay importable without it.

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
    from reachy_mini.utils import create_head_pose

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
    from reachy_mini import ReachyMini

    with ReachyMini(media_backend="default") as mini:
        pose = make_safe_pose(height, pitch)[0]
        mini.goto_target(pose, duration=MOVE_DURATION)


def click_labeled_points_with_review(
    frame,
    labels: list[str],
    window_name: str,
    polygon_order: list[str] | None = None,
    point_color: tuple[int, int, int] = (0, 0, 255),
    line_color: tuple[int, int, int] = (0, 255, 0),
) -> dict[str, list[int]] | None:
    """
    Collect labeled point clicks on a frozen frame, then show a review overlay.

    Returns a dict mapping label -> [x, y], or None if the user aborts.
    Loops until the user accepts or aborts, allowing retries.
    """
    order = polygon_order if polygon_order is not None else labels

    while True:
        points: dict[str, list[int]] = {}
        display = frame.copy()

        def mouse_callback(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            if len(points) >= len(labels):
                return
            label = labels[len(points)]
            points[label] = [x, y]
            cv2.circle(display, (x, y), 6, point_color, -1)
            cv2.putText(
                display, label, (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, point_color, 2,
            )
            print(f"  Clicked {label}: ({x}, {y})")
            cv2.imshow(window_name, display)

        print(f"\n{window_name}")
        print(f"Click in order: {', '.join(labels)}")
        print("ESC to abort.")

        cv2.namedWindow(window_name)
        cv2.imshow(window_name, display)
        cv2.setMouseCallback(window_name, mouse_callback)

        aborted = False
        while len(points) < len(labels):
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                aborted = True
                break

        cv2.setMouseCallback(window_name, lambda *a: None)

        if aborted:
            cv2.destroyWindow(window_name)
            return None

        # Draw review overlay: polygon lines + instructions
        review = display.copy()
        pts = [points[lbl] for lbl in order if lbl in points]
        for i in range(len(pts)):
            cv2.line(review, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), line_color, 2)
        cv2.putText(
            review,
            "ENTER/SPACE: accept   ESC/r: retry   q: abort",
            (10, review.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.imshow(window_name, review)
        print("Review: ENTER/SPACE to accept, ESC/r to retry, q to abort.")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, ord(" ")):  # ENTER or SPACE → accept
                cv2.destroyWindow(window_name)
                return points
            elif key in (27, ord("r")):  # ESC or r → retry
                print("Retrying...")
                break  # restart outer while loop
            elif key == ord("q"):  # q → abort
                cv2.destroyWindow(window_name)
                return None


def calibrate(setup_dir: Path = Path("data") / "raw_images") -> dict | None:
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
                if frame is None:
                    continue

                frozen_frame = frame.copy()
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

                actual_corners_px = click_labeled_points_with_review(
                    frozen_frame,
                    labels=["a1", "a8", "h8", "h1"],
                    window_name="Click actual board corners",
                    polygon_order=["a8", "h8", "h1", "a1"],
                )
                if actual_corners_px is None:
                    cv2.destroyAllWindows()
                    return None

                extended_corners_px = click_labeled_points_with_review(
                    frozen_frame,
                    labels=["a1", "a8", "h8", "h1"],
                    window_name="Click extended/padded board corners",
                    polygon_order=["a8", "h8", "h1", "a1"],
                    point_color=(255, 128, 0),
                    line_color=(0, 165, 255),
                )
                if extended_corners_px is None:
                    cv2.destroyAllWindows()
                    return None

                setup_dir.mkdir(parents=True, exist_ok=True)
                raw_image_path = setup_dir / "raw.png"
                metadata_path = setup_dir / "calibration_metadata.json"

                cv2.imwrite(str(raw_image_path), frozen_frame)

                calibration_data = {
                    "height_mm": last_sent_height,
                    "pitch_deg": last_sent_pitch,
                    "timestamp": timestamp,
                    "actual_corner_order": ["a1", "a8", "h8", "h1"],
                    "actual_corners_px": actual_corners_px,
                    "camera_natural_orientation": infer_camera_natural_corner_order(actual_corners_px),
                    "extended_corner_order": ["a1", "a8", "h8", "h1"],
                    "extended_corners_px": extended_corners_px,
                    "raw_image_path": str(raw_image_path),
                }

                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(calibration_data, f, indent=2)

                print(f"Saved raw image: {raw_image_path}")
                print(f"Saved calibration metadata: {metadata_path}")

                cv2.destroyAllWindows()
                return calibration_data

            elif key == ord("q"):
                cv2.destroyAllWindows()
                return None
            else:
                continue

            # Clamp and move robot
            pose, safe_height, safe_pitch = make_safe_pose(new_height_mm, new_pitch_deg)

            if safe_height != new_height_mm or safe_pitch != new_pitch_deg:
                print(
                    "Requested pose outside safe range. "
                    f"Clamped to height={safe_height}, pitch={safe_pitch}"
                )

            if safe_height != last_sent_height or safe_pitch != last_sent_pitch:
                print(f"Moving to height={safe_height}, pitch={safe_pitch}")
                try:
                    mini.set_target(head=pose, body_yaw=None)
                    height_mm = safe_height
                    pitch_deg = safe_pitch
                    last_sent_height = safe_height
                    last_sent_pitch = safe_pitch
                    time.sleep(MOVE_DURATION)
                except Exception as e:
                    print("Move failed:", e)
                    print(f"Keeping previous pose: height={height_mm}, pitch={pitch_deg}")


def infer_camera_natural_corner_order(corners_px: dict[str, list[int]]) -> dict:
    """
    Infer which semantic board corner sits in each visual quadrant (top-left,
    top-right, bottom-right, bottom-left) using pixel-coordinate comparisons.

    Tries all four cyclic orderings of the four corners and scores each by
    counting how many expected image-coordinate inequalities hold.
    Smaller x → further left; smaller y → higher up (image convention).
    """
    candidates = [
        ["a8", "h8", "h1", "a1"],
        ["h8", "h1", "a1", "a8"],
        ["h1", "a1", "a8", "h8"],
        ["a1", "a8", "h8", "h1"],
    ]

    def x(label: str) -> int:
        return corners_px[label][0]

    def y(label: str) -> int:
        return corners_px[label][1]

    scored = []
    for candidate in candidates:
        tl, tr, br, bl = candidate
        score = sum([
            x(tl) < x(tr),
            x(bl) < x(br),
            y(tl) < y(bl),
            y(tr) < y(br),
            x(tl) < x(br),
            x(bl) < x(tr),
            y(tl) < y(br),
            y(tr) < y(bl),
        ])
        scored.append({"order": candidate, "score": score})

    scored.sort(key=lambda e: e["score"], reverse=True)
    best = scored[0]
    tl, tr, br, bl = best["order"]

    return {
        "order": {
            "tl": tl,
            "tr": tr,
            "br": br,
            "bl": bl,
        },
        "score": best["score"],
        "all_scores": scored,
        "ambiguous": scored[0]["score"] == scored[1]["score"],
    }


LABEL_ORDER = ["a1", "a8", "h8", "h1"]


def derive_center_px(actual_corners_px: dict, order: dict, board_size: int = 400) -> list:
    """Camera-pixel position of the board centre (board coord (0.5, 0.5)), derived from H.

    ``order`` maps ``tl/tr/br/bl`` to the corresponding square label (from
    ``camera_natural_orientation``). Derived from the homography rather than clicked, which is
    more precise than any hand click at the centre.
    """
    last = board_size - 1
    src = np.array(
        [actual_corners_px[order[pos]] for pos in ["tl", "tr", "br", "bl"]], dtype=np.float32
    )
    dst = np.array([[0, 0], [last, 0], [last, last], [0, last]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    center_warped = np.array([[[last / 2.0, last / 2.0]]], dtype=np.float32)
    center_px = cv2.perspectiveTransform(center_warped, np.linalg.inv(homography)).reshape(2)
    return [float(center_px[0]), float(center_px[1])]


def build_calibration_metadata(
    *,
    existing: dict,
    actual_corners_px: dict,
    extended_corners_px: dict,
    extended_center_px,
    K,
    D,
    image_size,
    board_size: int = 400,
    raw_image_path=None,
) -> dict:
    """Assemble versioned (v2) calibration metadata, preserving any pre-existing fields.

    Mirrors the ``annotate_existing`` idiom: spread ``**existing`` first, then add/override
    only the new fields. ``center_px`` is derived from the homography (not a click); the scaled
    camera intrinsics are cached so the batch/gameplay never re-fetch them from reachy_mini.
    """
    order_info = infer_camera_natural_corner_order(actual_corners_px)
    metadata = {
        **existing,
        "calibration_version": 2,
        "actual_corner_order": LABEL_ORDER,
        "actual_corners_px": actual_corners_px,
        "camera_natural_orientation": order_info,
        "extended_corner_order": LABEL_ORDER,
        "extended_corners_px": extended_corners_px,
        "extended_center_px": list(extended_center_px),
        "center_px": derive_center_px(actual_corners_px, order_info["order"], board_size),
        "camera_intrinsics": {
            "K": np.asarray(K, dtype=float).tolist(),
            "D": np.asarray(D, dtype=float).reshape(-1).tolist(),
            "image_size": [int(image_size[0]), int(image_size[1])],
        },
    }
    if raw_image_path is not None:
        metadata["raw_image_path"] = str(raw_image_path)
    return metadata


def annotate_existing(image_path: Path) -> dict | None:
    """
    Load a stored raw image, collect corner annotations interactively, and
    write (or overwrite) calibration_metadata.json in the same directory.

    Existing metadata fields (height_mm, pitch_deg, timestamp, …) are
    preserved; only corner fields are replaced.

    Usage:
        python -m chess_assistant.calibration data/2026-06-17_210454/raw.png
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Failed to load image: {image_path}")
        return None

    setup_dir = image_path.parent
    metadata_path = setup_dir / "calibration_metadata.json"

    existing: dict = {}
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded existing metadata from: {metadata_path}")

    actual_corners_px = click_labeled_points_with_review(
        frame,
        labels=["a1", "a8", "h8", "h1"],
        window_name="Click actual board corners",
        polygon_order=["a8", "h8", "h1", "a1"],
    )
    if actual_corners_px is None:
        cv2.destroyAllWindows()
        return None

    extended_corners_px = click_labeled_points_with_review(
        frame,
        labels=["a1", "a8", "h8", "h1"],
        window_name="Click extended/padded board corners",
        polygon_order=["a8", "h8", "h1", "a1"],
        point_color=(255, 128, 0),
        line_color=(0, 165, 255),
    )
    if extended_corners_px is None:
        cv2.destroyAllWindows()
        return None

    calibration_data = {
        **existing,
        "actual_corner_order": ["a1", "a8", "h8", "h1"],
        "actual_corners_px": actual_corners_px,
        "camera_natural_orientation": infer_camera_natural_corner_order(actual_corners_px),
        "extended_corner_order": ["a1", "a8", "h8", "h1"],
        "extended_corners_px": extended_corners_px,
        "raw_image_path": str(image_path),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"Saved calibration metadata: {metadata_path}")
    cv2.destroyAllWindows()
    return calibration_data


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        annotate_existing(Path(sys.argv[1]))
    else:
        calibrate()
