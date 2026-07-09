import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from chess_assistant.camera_utils import build_undistort_maps, get_lite_camera_KD, undistort
from chess_assistant.config import SQUARES

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


def _log_calibration_summary(calibration_data: dict, config_path) -> None:
    """Build a Processor from the calibration and log the vanishing-point residual (§8)."""
    try:
        from chess_assistant.image_processing import Processor

        processor = Processor(calibration_data, config_path)
        print(f"Vanishing-point residual: {processor.vp_residual:.3f} px")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not compute calibration summary: {exc})")


def calibrate(
    setup_dir: Path = Path("data") / "raw_images",
    config_path="config.yaml",
) -> dict | None:
    from reachy_mini import ReachyMini

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

                setup_dir.mkdir(parents=True, exist_ok=True)
                raw_image_path = setup_dir / "raw.png"
                metadata_path = setup_dir / "calibration_metadata.json"

                # Save the original distorted capture untouched, then do all clicks + geometry
                # on the undistorted frame.
                cv2.imwrite(str(raw_image_path), frozen_frame)
                undistorted, K, D, image_size = undistort_reference_frame(frozen_frame, setup_dir)
                cv2.destroyWindow("Reachy board view")

                collected = CalibrationUI(undistorted, config_path).run()
                if collected is None:
                    cv2.destroyAllWindows()
                    return None
                base_points, extended_points = collected

                calibration_data = build_calibration_metadata(
                    existing={
                        "height_mm": last_sent_height,
                        "pitch_deg": last_sent_pitch,
                        "timestamp": timestamp,
                    },
                    actual_corners_px=base_points,
                    extended_corners_px={k: v for k, v in extended_points.items() if k != "center"},
                    extended_center_px=extended_points["center"],
                    K=K,
                    D=D,
                    image_size=image_size,
                    raw_image_path=raw_image_path,
                )

                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(calibration_data, f, indent=2)

                _log_calibration_summary(calibration_data, config_path)
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


def undistort_reference_frame(frame, setup_dir: Path):
    """Undistort a setup's reference frame in memory and persist it for later visual diffing.

    Returns ``(undistorted, K, D, image_size)`` and writes ``raw_undistorted.png`` next to
    ``raw.png`` (straight real-world edges should look straighter there). The original
    distorted ``raw.png`` is written/kept by the caller and never modified here.
    """
    height, width = frame.shape[:2]
    image_size = (width, height)
    K, D = get_lite_camera_KD(image_size)
    map1, map2 = build_undistort_maps(K, D, image_size)
    undistorted = undistort(frame, map1, map2)
    cv2.imwrite(str(setup_dir / "raw_undistorted.png"), undistorted)
    return undistorted, K, D, image_size


# --------------------------------------------------------------------------------------------
# Interactive calibration UI (runs on the undistorted frame): 5-point collection, a quick
# review with drag-to-correct, and a background per-square mask inspector.
# --------------------------------------------------------------------------------------------

BASE_CORNER_LABELS = ["a1", "a8", "h8", "h1"]
EXTENDED_LABELS = ["a1", "a8", "h8", "h1", "center"]

_BASE_COLOR = (0, 0, 255)           # red   - clicked base corners
_CENTER_BASE_COLOR = (0, 255, 255)  # yellow - derived centre base (not clicked)
_EXTENDED_COLOR = (255, 128, 0)     # orange - clicked extended points
_LINK_COLOR = (0, 255, 0)           # green  - base->extended links
_CEILING_COLOR = (0, 200, 200)      # cyan   - ceiling quad / cuboid ceiling
_WALL_COLOR = (200, 200, 0)         # cuboid walls
_DRAG_RADIUS = 14


@dataclass
class SquareResult:
    """One square's inspector view: crop pixels + cuboid corners in crop-local coordinates."""

    label: str
    crop: np.ndarray
    floor_local: np.ndarray    # (4, 2)
    ceiling_local: np.ndarray  # (4, 2)


def build_inspector_calibration(base_points: dict, extended_points: dict) -> dict:
    """Minimal geometry-only calibration dict from in-progress clicks, for the inspector."""
    return {
        "camera_natural_orientation": infer_camera_natural_corner_order(base_points),
        "actual_corners_px": base_points,
        "extended_corners_px": {k: v for k, v in extended_points.items() if k != "center"},
        "extended_center_px": extended_points["center"],
    }


def compute_inspector_results(frame, calibration_dict, config_path="config.yaml") -> list:
    """Warp the (undistorted) frame and build a SquareResult per square (64), in SQUARES order.

    Pure/headless (no cv2 GUI); runs in a background thread during the review step.
    """
    from chess_assistant.image_processing import Processor

    processor = Processor(calibration_dict, config_path)
    warped = cv2.warpPerspective(frame, processor.matrix, processor.image_size)
    results = []
    for label in SQUARES:
        geom = processor.square_geometry[label]
        x_min, y_min, x_max, y_max = geom.bbox
        offset = np.array([x_min, y_min])
        results.append(
            SquareResult(
                label=label,
                crop=warped[y_min:y_max, x_min:x_max].copy(),
                floor_local=geom.floor_pts - offset,
                ceiling_local=geom.ceiling_pts - offset,
            )
        )
    return results


def _draw_cuboid(canvas, floor, ceiling):
    """Thin cuboid wireframe (floor quad + ceiling quad + 4 walls), no fill."""
    floor = np.asarray(floor).astype(np.int32)
    ceiling = np.asarray(ceiling).astype(np.int32)
    cv2.polylines(canvas, [floor], True, _BASE_COLOR, 1)
    cv2.polylines(canvas, [ceiling], True, _CEILING_COLOR, 1)
    for f, c in zip(floor, ceiling):
        cv2.line(canvas, tuple(f), tuple(c), _WALL_COLOR, 1)
    return canvas


def render_square_inspector(result: SquareResult, size: int = 320) -> np.ndarray:
    """Upscale a square's crop and draw its cuboid wireframe on top (crop pixels stay visible)."""
    crop = result.crop
    if crop is None or crop.size == 0:
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
    scale = size / max(crop.shape[0], crop.shape[1])
    disp = cv2.resize(
        crop,
        (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    _draw_cuboid(disp, result.floor_local * scale, result.ceiling_local * scale)
    cv2.putText(disp, result.label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return disp


def render_review_overlay(frame, base_points, center_base, extended_points) -> np.ndarray:
    """Quick review overlay: 10 points, base->extended links, and the ceiling quad."""
    img = frame.copy()

    def pt(p):
        return (int(round(p[0])), int(round(p[1])))

    for label in BASE_CORNER_LABELS:
        if label in base_points and label in extended_points:
            cv2.line(img, pt(base_points[label]), pt(extended_points[label]), _LINK_COLOR, 1)
    if center_base is not None and "center" in extended_points:
        cv2.line(img, pt(center_base), pt(extended_points["center"]), _LINK_COLOR, 1)

    if len(base_points) == 4:
        order = infer_camera_natural_corner_order(base_points)["order"]
        if all(order[p] in extended_points for p in ["tl", "tr", "br", "bl"]):
            ceiling = np.array(
                [extended_points[order[p]] for p in ["tl", "tr", "br", "bl"]], dtype=np.int32
            )
            cv2.polylines(img, [ceiling], True, _CEILING_COLOR, 1)

    for p in base_points.values():
        cv2.circle(img, pt(p), 5, _BASE_COLOR, -1)
    if center_base is not None:
        cv2.circle(img, pt(center_base), 5, _CENTER_BASE_COLOR, -1)
    for p in extended_points.values():
        cv2.circle(img, pt(p), 5, _EXTENDED_COLOR, -1)
    return img


class CalibrationUI:
    """Interactive 5-point calibration collector on an already-undistorted frame.

    Flow: click 4 base corners -> click 5 extended points (4 corners + centre) -> a review step
    with drag-to-correct and a background per-square mask inspector. ``run()`` returns
    ``(base_points, extended_points)`` or ``None`` on abort.
    """

    def __init__(self, frame, config_path="config.yaml", window_name="Calibration"):
        self.frame = frame
        self.config_path = config_path
        self.window_name = window_name
        self.base_points: dict[str, list[int]] = {}
        self.extended_points: dict[str, list[int]] = {}
        self.selected = None
        self.generation = 0
        self.results: list = [None] * 64
        self.results_lock = threading.Lock()
        self.inspector_index = 0
        self.show_inspector = False

    def run(self):
        while True:
            self.base_points, self.extended_points = {}, {}
            self.show_inspector = False
            if not self._collect(self.base_points, BASE_CORNER_LABELS, "actual board corners", _BASE_COLOR):
                cv2.destroyWindow(self.window_name)
                return None
            if not self._collect(
                self.extended_points, EXTENDED_LABELS, "extended points (corners then centre)", _EXTENDED_COLOR
            ):
                cv2.destroyWindow(self.window_name)
                return None
            outcome = self._review()
            if outcome == "retry":
                continue
            cv2.destroyWindow(self.window_name)
            return outcome  # (base_points, extended_points) or None

    def _collect(self, store, labels, title, color) -> bool:
        display = self.frame.copy()
        store.clear()

        def on_click(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN or len(store) >= len(labels):
                return
            label = labels[len(store)]
            store[label] = [x, y]
            cv2.circle(display, (x, y), 6, color, -1)
            cv2.putText(display, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(self.window_name, display)

        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, display)
        cv2.setMouseCallback(self.window_name, on_click)
        print(f"\nClick {title} in order: {', '.join(labels)}. ESC to abort.")
        while len(store) < len(labels):
            if (cv2.waitKey(20) & 0xFF) == 27:  # ESC
                cv2.setMouseCallback(self.window_name, lambda *a: None)
                return False
        cv2.setMouseCallback(self.window_name, lambda *a: None)
        return True

    def _center_base(self):
        order = infer_camera_natural_corner_order(self.base_points)["order"]
        return derive_center_px(self.base_points, order)

    def _launch_inspector(self):
        self.generation += 1
        generation = self.generation
        frame = self.frame.copy()
        calibration = build_inspector_calibration(self.base_points, self.extended_points)
        config_path = self.config_path
        with self.results_lock:
            self.results = [None] * 64

        def work():
            try:
                results = compute_inspector_results(frame, calibration, config_path)
            except Exception as exc:  # noqa: BLE001 - a bad in-progress click shouldn't crash the UI
                print(f"inspector computation failed: {exc}")
                return
            with self.results_lock:
                if generation == self.generation:  # discard stale computations
                    self.results = results

        threading.Thread(target=work, daemon=True).start()

    def _find_marker(self, x, y):
        best, best_dist = None, _DRAG_RADIUS ** 2
        markers = [("base", lbl, p) for lbl, p in self.base_points.items()]
        markers += [("extended", lbl, p) for lbl, p in self.extended_points.items()]
        for which, label, p in markers:
            dist = (p[0] - x) ** 2 + (p[1] - y) ** 2
            if dist <= best_dist:
                best, best_dist = (which, label), dist
        return best

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected = self._find_marker(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.selected is not None:
            which, label = self.selected
            store = self.base_points if which == "base" else self.extended_points
            store[label] = [x, y]
            self._redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selected is not None:
                self.selected = None
                self._launch_inspector()  # recompute from the corrected points
        elif event == cv2.EVENT_MOUSEWHEEL and self.show_inspector:
            step = 1 if cv2.getMouseWheelDelta(flags) > 0 else -1
            self.inspector_index = (self.inspector_index + step) % 64
            self._redraw()

    def _redraw(self):
        if self.show_inspector:
            with self.results_lock:
                result = self.results[self.inspector_index]
            if result is None:
                view = np.zeros((320, 320, 3), dtype=np.uint8)
                cv2.putText(
                    view, f"{SQUARES[self.inspector_index]}: computing...",
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
                )
            else:
                view = render_square_inspector(result)
            cv2.imshow(self.window_name, view)
        else:
            cv2.imshow(
                self.window_name,
                render_review_overlay(
                    self.frame, self.base_points, self._center_base(), self.extended_points
                ),
            )

    def _review(self):
        self._launch_inspector()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._redraw()
        print(
            "Review: drag any marker to correct. 'i' toggles the per-square inspector, "
            "mouse-wheel scrolls squares. ENTER/SPACE accept, 'r' retry, 'q'/ESC abort."
        )
        while True:
            key = cv2.waitKey(30) & 0xFF
            if self.show_inspector:
                self._redraw()  # pick up freshly computed inspector results
            if key in (13, ord(" ")):
                cv2.setMouseCallback(self.window_name, lambda *a: None)
                return self.base_points, self.extended_points
            if key == ord("i"):
                self.show_inspector = not self.show_inspector
                self._redraw()
            elif key == ord("r"):
                cv2.setMouseCallback(self.window_name, lambda *a: None)
                return "retry"
            elif key in (ord("q"), 27):
                cv2.setMouseCallback(self.window_name, lambda *a: None)
                return None


def annotate_existing(image_path: Path, config_path="config.yaml") -> dict | None:
    """
    Load a stored raw image, undistort it, collect the v2 calibration (4 base corners + 5
    extended points including the centre) interactively, and write (or overwrite)
    calibration_metadata.json in the same directory.

    Pre-existing metadata fields (height_mm, pitch_deg, timestamp, …) are preserved; the
    corner/centre/intrinsics fields are (re)written on the undistorted frame.

    Usage:
        python -m chess_assistant.calibration data/generated/<setup>/raw.png
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

    undistorted, K, D, image_size = undistort_reference_frame(frame, setup_dir)

    collected = CalibrationUI(undistorted, config_path).run()
    if collected is None:
        cv2.destroyAllWindows()
        return None
    base_points, extended_points = collected

    calibration_data = build_calibration_metadata(
        existing=existing,
        actual_corners_px=base_points,
        extended_corners_px={k: v for k, v in extended_points.items() if k != "center"},
        extended_center_px=extended_points["center"],
        K=K,
        D=D,
        image_size=image_size,
        raw_image_path=image_path,
    )

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2)

    _log_calibration_summary(calibration_data, config_path)
    print(f"Saved calibration metadata: {metadata_path}")
    cv2.destroyAllWindows()
    return calibration_data


def relabel_existing_setups(
    data_root: Path = Path("data") / "generated",
    config_path="config.yaml",
) -> None:
    """Phase 1: re-run the v2 calibration UI over every existing setup's stored ``raw.png``.

    Opens each ``<setup>/raw.png`` undistorted in the improved UI (re-clicking all corners plus
    the new centre fresh) and writes updated versioned metadata in place. Only touches the
    setups, not the captured frames — Phase 2 (``regenerate.py``) regenerates those afterwards.
    """
    data_root = Path(data_root)
    setups = sorted(
        p for p in data_root.iterdir() if p.is_dir() and (p / "raw.png").exists()
    )
    print(f"Relabelling {len(setups)} setups under {data_root}.")
    print("Per setup: ENTER/SPACE accept, 'r' retry, 'q'/ESC abort (stops the session).")
    for i, setup_dir in enumerate(setups, 1):
        print(f"\n[{i}/{len(setups)}] {setup_dir.name}")
        if annotate_existing(setup_dir / "raw.png", config_path) is None:
            print("Aborted; stopping relabelling session.")
            break


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "relabel":
            relabel_existing_setups()
        else:
            annotate_existing(Path(sys.argv[1]))
    else:
        calibrate()
