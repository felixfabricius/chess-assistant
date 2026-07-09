"""Offline batch regeneration of warped images + per-square crops (Phase 2).

Two-phase workflow:

* **Phase 1 — relabelling** (``chess_assistant.calibration.relabel_existing_setups``): the
  interactive session that opens each existing setup's ``raw.png`` undistorted and writes an
  updated, versioned ``calibration_metadata.json`` (adds the centre point + camera intrinsics).
* **Phase 2 — regeneration** (this module): for each setup, build its :class:`Processor` once
  (freezing the undistortion maps, homography, vanishing point, magnitude field and all 64
  square geometries) and reuse it across every frame; per frame only undistort -> warp ->
  cutout -> merge the preserved label back in.

**Labels are never derivable from pixels** — they were written by the live capture session and
live only in each square's ``_metadata.json`` (and the CSV). So each square's existing
``"label"`` is read *before* cutout overwrites its metadata, then merged back afterwards.

Reads camera intrinsics from each setup's metadata, so the batch never imports ``reachy_mini``
and the worker is a plain top-level function (importable by name under Windows "spawn").
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from chess_assistant.image_processing import Processor

DATA_ROOT = Path("data") / "generated"

# Per-worker-process cache: one Processor per setup, built the first time a worker sees the
# setup and reused for all of that setup's frames handled by the same worker.
_PROCESSOR_CACHE: dict[str, Processor] = {}


def _get_processor(setup_dir: Path, config_path: str | None) -> Processor:
    key = str(setup_dir)
    processor = _PROCESSOR_CACHE.get(key)
    if processor is None:
        processor = Processor(setup_dir / "calibration_metadata.json", config_path)
        _PROCESSOR_CACHE[key] = processor
    return processor


def read_existing_labels(squares_dir: Path) -> dict[str, str]:
    """Collect the ``"label"`` field from every existing per-square metadata JSON."""
    labels: dict[str, str] = {}
    if not squares_dir.exists():
        return labels
    for square_dir in squares_dir.iterdir():
        if not square_dir.is_dir():
            continue
        square = square_dir.name
        meta_path = square_dir / f"{square}_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "label" in meta:
                labels[square] = meta["label"]
    return labels


def merge_preserved_labels(squares_dir: Path, labels: dict[str, str]) -> None:
    """Re-attach preserved labels to the freshly written per-square metadata JSONs."""
    for square, label in labels.items():
        meta_path = squares_dir / square / f"{square}_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["label"] = label
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def regenerate_frame(setup_dir_str: str, frame_dir_str: str, config_path_str: str | None) -> str:
    """Regenerate one frame's warped image + square crops in place, preserving labels.

    Top-level and picklable so it can be submitted to a ``ProcessPoolExecutor`` under spawn.
    """
    setup_dir = Path(setup_dir_str)
    frame_dir = Path(frame_dir_str)
    squares_dir = frame_dir / "squares"

    # Capture the ground-truth labels BEFORE cutout overwrites the per-square metadata.
    preserved_labels = read_existing_labels(squares_dir)

    processor = _get_processor(setup_dir, config_path_str)
    warped_path = processor.warp(frame_dir / "image.png")
    processor.cutout(warped_path)

    merge_preserved_labels(squares_dir, preserved_labels)
    return frame_dir_str


def iter_frames(data_root: Path):
    """Yield ``(setup_dir, frame_dir)`` for every regenerable frame under ``data_root``."""
    for setup_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if not (setup_dir / "calibration_metadata.json").exists():
            continue
        for frame_dir in sorted(setup_dir.glob("board_*")):
            if (frame_dir / "image.png").exists():
                yield setup_dir, frame_dir


def regenerate_all(data_root=DATA_ROOT, config_path="config.yaml", max_workers=None) -> None:
    """Regenerate every frame under ``data_root`` using one persistent process pool."""
    tasks = list(iter_frames(Path(data_root)))
    print(f"Regenerating {len(tasks)} frames...")
    failures = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(regenerate_frame, str(setup), str(frame), str(config_path)): frame
            for setup, frame in tasks
        }
        for future in as_completed(futures):
            frame = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - report and keep going
                failures += 1
                print(f"FAILED {frame}: {exc}")
    print(f"Done. {len(tasks) - failures}/{len(tasks)} frames regenerated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    regenerate_all(args.data_root, args.config, args.workers)
