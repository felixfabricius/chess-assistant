"""Data-generation tool for the chess vision classifier.

Workflow (see README / module docstring at bottom for the full write-up):

1. A ``pygame`` window shows a virtual chessboard (White at the bottom).
2. You make a move / edit the position on the virtual board.
3. You make the *same* change on the real physical board.
4. You press Space; the robot captures a photo.
5. The photo is warped and cut into 64 square cutouts by the existing
   :class:`~chess_assistant.image_processing.Processor`.
6. The virtual board is the source of truth for the label of every square.
7. Labels are written into each square's metadata JSON, a per-image
   ``metadata.json`` is saved, and 64 rows are appended to the master CSV.

The pure helper functions and :class:`DataGenerationSession` contain no
``pygame`` / robot code so they can be unit tested without a display.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import chess

from chess_assistant.calibration import calibrate
from chess_assistant.camera import capture_image
from chess_assistant.config import SQUARES
from chess_assistant.image_processing import Processor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Root under which everything is stored, grouped by physical setup.
DATA_ROOT = Path("data") / "generated"
CSV_NAME = "data.csv"

# One row per square cutout. Kept as a stable, ordered schema so the CSV can be
# appended to over many sessions without ever changing column order.
CSV_COLUMNS: list[str] = [
    "setup_id",
    "image_id",
    "square",
    "label",
    "square_image_path",
    "full_image_path",
    "calibration_metadata_path",
    "valid_game_position",
    "board_fen",
    "previous_board_fen",
    "move_uci",
    "created_at",
]

# 13-class label for an empty square.
EMPTY_LABEL = "empty"

# Piece letters accepted for spawning in free-placement mode (uppercase = white,
# lowercase = black), and the promotion choices in legal-move mode.
PIECE_SYMBOLS = set("PNBRQKpnbrqk")
PROMOTION_MAP = {
    "q": chess.QUEEN,
    "r": chess.ROOK,
    "b": chess.BISHOP,
    "n": chess.KNIGHT,
}


# ---------------------------------------------------------------------------
# Pure helpers (no pygame / robot dependencies -> easy to unit test)
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Return the current time as an ISO-8601 string (seconds resolution)."""
    return datetime.now().isoformat(timespec="seconds")


def now_stamp() -> str:
    """Return a filesystem-friendly timestamp, e.g. ``2026-07-01_143512``."""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def save_json(path: Path, data: dict) -> None:
    """Write ``data`` as indented UTF-8 JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def piece_label_at(board: chess.Board, square: str) -> str:
    """Return the 13-class label of ``square`` (e.g. ``"e4"``).

    ``"empty"`` if there is no piece, otherwise the piece symbol
    (``"P"``, ``"n"``, ...). The virtual board is the source of truth.
    """
    piece = board.piece_at(chess.parse_square(square))
    return piece.symbol() if piece is not None else EMPTY_LABEL


def board_to_piece_map(board: chess.Board) -> dict[str, str]:
    """Map every one of the 64 squares to its 13-class label."""
    return {square: piece_label_at(board, square) for square in SQUARES}


def square_image_path(squares_dir: Path, square: str) -> Path:
    """Path to the cutout image, e.g. ``squares/e4/e4.png``.

    Matches the nested structure produced by ``Processor.cutout``.
    """
    return Path(squares_dir) / square / f"{square}.png"


def square_annotated_image_path(squares_dir: Path, square: str) -> Path:
    """Path to the annotated cutout, e.g. ``squares/e4/e4_annotated.png``."""
    return Path(squares_dir) / square / f"{square}_annotated.png"


def build_square_rows(
    *,
    setup_id: str,
    image_id: str,
    squares_dir: Path,
    full_image_path: Path,
    calibration_metadata_path: Path,
    piece_map: dict[str, str],
    valid_game_position: bool,
    board_fen: str | None,
    previous_board_fen: str | None,
    move_uci: str | None,
    created_at: str,
    squares: list[str] = SQUARES,
) -> list[dict]:
    """Build the 64 CSV rows (one per square) for a single captured image."""
    rows: list[dict] = []
    for square in squares:
        rows.append(
            {
                "setup_id": setup_id,
                "image_id": image_id,
                "square": square,
                "label": piece_map[square],
                "square_image_path": str(square_image_path(squares_dir, square)),
                "full_image_path": str(full_image_path),
                "calibration_metadata_path": str(calibration_metadata_path),
                "valid_game_position": valid_game_position,
                "board_fen": board_fen or "",
                "previous_board_fen": previous_board_fen or "",
                "move_uci": move_uci or "",
                "created_at": created_at,
            }
        )
    return rows


def append_rows_to_csv(
    csv_path: Path,
    rows: list[dict],
    columns: list[str] = CSV_COLUMNS,
) -> None:
    """Append ``rows`` to the master CSV, writing a header if the file is new.

    Uses append mode so existing data is never lost and a crash mid-write can
    at worst leave a truncated final row rather than corrupting the whole file.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def create_setup(data_root: Path, timestamp: str | None = None) -> tuple[str, Path]:
    """Create (and return) a new timestamped setup directory.

    Returns ``(setup_id, setup_dir)``. ``timestamp`` can be injected for tests.
    """
    setup_id = timestamp or now_stamp()
    setup_dir = Path(data_root) / setup_id
    setup_dir.mkdir(parents=True, exist_ok=True)
    return setup_id, setup_dir


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class DataGenerationSession:
    """Holds all mutable state for a data-generation run.

    Intentionally free of any ``pygame`` / OpenCV window code; the GUI
    (:class:`BoardUI`) drives this object, and the robot pipeline
    (calibrate / capture / warp / cutout) is called from here.
    """

    def __init__(self, config_path: Path, data_root: Path = DATA_ROOT) -> None:
        self.config_path = Path(config_path)
        self.data_root = Path(data_root)
        self.csv_path = self.data_root / CSV_NAME

        # Set once a setup has been calibrated.
        self.setup_id: str | None = None
        self.setup_dir: Path | None = None
        self.calibration_metadata_path: Path | None = None
        self.processor: Processor | None = None

        # Virtual-board state.
        self.board = chess.Board()
        self.legal_mode = True
        self.valid_game_position = True
        self.previous_board_fen: str | None = None
        self.board_fen: str | None = self.board.fen()
        self.move_uci: str | None = None

        self.image_count = 0

    # -- setup / calibration ------------------------------------------------

    def start_new_setup(self) -> bool:
        """Create a new setup dir, calibrate, and (re)initialise the Processor.

        Resets the virtual board to a fresh game. Returns ``True`` on success.
        """
        setup_id, setup_dir = create_setup(self.data_root)

        calibration_data = calibrate(setup_dir)
        if not calibration_data:
            print("Calibration was aborted or failed. Setup not created.")
            return False

        calibration_metadata_path = setup_dir / "calibration_metadata.json"
        try:
            processor = Processor(calibration_metadata_path, self.config_path)
        except Exception as exc:  # noqa: BLE001 - surface any init failure
            print(f"Failed to initialise Processor: {exc}")
            return False

        self.setup_id = setup_id
        self.setup_dir = setup_dir
        self.calibration_metadata_path = calibration_metadata_path
        self.processor = processor

        save_json(
            setup_dir / "setup_metadata.json",
            {
                "setup_id": setup_id,
                "created_at": now_iso(),
                "config_path": str(self.config_path),
                "calibration_metadata_path": str(calibration_metadata_path),
            },
        )

        self.new_game()
        print(f"New setup ready: {setup_id}")
        return True

    # -- virtual board: legal-move mode ------------------------------------

    def new_game(self) -> None:
        """Reset to the standard starting position and legal-move mode."""
        self.board = chess.Board()
        self.legal_mode = True
        self.valid_game_position = True
        self.previous_board_fen = None
        self.board_fen = self.board.fen()
        self.move_uci = None

    def apply_legal_move(
        self,
        from_square: str,
        to_square: str,
        promotion: int | None = None,
    ) -> str:
        """Attempt a legal move from ``from_square`` to ``to_square``.

        Returns one of:
        - ``"ok"``            move applied; FEN tracking updated.
        - ``"illegal"``       no such legal move.
        - ``"need_promotion"``the move is a pawn promotion; call again with
                              ``promotion`` set to a piece type.
        """
        from_idx = chess.parse_square(from_square)
        to_idx = chess.parse_square(to_square)

        try:
            candidates = [
                m
                for m in self.board.legal_moves
                if m.from_square == from_idx and m.to_square == to_idx
            ]
        except Exception:  # noqa: BLE001 - board may be odd after free edits
            return "illegal"

        if not candidates:
            return "illegal"

        if promotion is None:
            needs_promotion = any(m.promotion is not None for m in candidates)
            if needs_promotion and len(candidates) > 1:
                return "need_promotion"
            move = candidates[0]
        else:
            move = next((m for m in candidates if m.promotion == promotion), None)
            if move is None:
                return "illegal"

        self.previous_board_fen = self.board.fen()
        self.board.push(move)
        self.board_fen = self.board.fen()
        self.move_uci = move.uci()
        return "ok"

    # -- virtual board: free-placement mode --------------------------------

    def toggle_mode(self) -> None:
        """Toggle between legal-move and free-placement mode."""
        self.legal_mode = not self.legal_mode

    def _mark_edited(self) -> None:
        """Record that the position was edited freely (no longer a valid game)."""
        self.valid_game_position = False
        self.move_uci = None
        self.previous_board_fen = None
        self.board_fen = self._current_fen()

    def place_piece(self, square: str, symbol: str) -> None:
        """Place ``symbol`` (e.g. ``"Q"``/``"q"``) on ``square``, ignoring legality."""
        self.board.set_piece_at(
            chess.parse_square(square), chess.Piece.from_symbol(symbol)
        )
        self._mark_edited()

    def remove_piece(self, square: str) -> None:
        """Remove any piece from ``square``."""
        self.board.remove_piece_at(chess.parse_square(square))
        self._mark_edited()

    def move_piece_free(self, from_square: str, to_square: str) -> bool:
        """Move whatever is on ``from_square`` to ``to_square``, ignoring legality."""
        from_idx = chess.parse_square(from_square)
        piece = self.board.piece_at(from_idx)
        if piece is None:
            return False
        self.board.remove_piece_at(from_idx)
        self.board.set_piece_at(chess.parse_square(to_square), piece)
        self._mark_edited()
        return True

    def _current_fen(self) -> str | None:
        """Best-effort FEN; ``None`` if the position cannot be represented."""
        try:
            return self.board.fen()
        except Exception:  # noqa: BLE001
            return None

    # -- capture ------------------------------------------------------------

    def capture(self) -> bool:
        """Capture a photo, warp+cutout it, write labels and append CSV rows.

        Returns ``True`` only if the full pipeline succeeded. On any failure
        (capture, warp, cutout) nothing is appended to the CSV.
        """
        if self.processor is None or self.setup_dir is None:
            print("No active setup. Press 'r' to calibrate a setup first.")
            return False

        created_at = now_iso()

        # 1. Capture. capture_image() creates its own board_<timestamp> dir.
        try:
            image_dir = capture_image(self.setup_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"Image capture failed: {exc}")
            return False

        full_image_path = image_dir / "image.png"

        # 2. Warp + cutout using the existing Processor.
        try:
            warped_image_path = self.processor.warp(full_image_path)
            squares_dir = self.processor.cutout(warped_image_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Warp/cutout failed: {exc}. No rows written.")
            return False

        image_id = image_dir.name
        piece_map = board_to_piece_map(self.board)
        board_fen = self.board_fen

        # 3. Append the label to each square's existing metadata JSON.
        for square in SQUARES:
            meta_path = squares_dir / square / f"{square}_metadata.json"
            existing: dict = {}
            if meta_path.exists():
                try:
                    with meta_path.open(encoding="utf-8") as f:
                        existing = json.load(f)
                except (OSError, json.JSONDecodeError):
                    existing = {}
            existing["label"] = piece_map[square]
            save_json(meta_path, existing)

        # 4. Append the 64 CSV rows (built first, then written in one go).
        rows = build_square_rows(
            setup_id=self.setup_id,
            image_id=image_id,
            squares_dir=squares_dir,
            full_image_path=full_image_path,
            calibration_metadata_path=self.calibration_metadata_path,
            piece_map=piece_map,
            valid_game_position=self.valid_game_position,
            board_fen=board_fen,
            previous_board_fen=self.previous_board_fen,
            move_uci=self.move_uci,
            created_at=created_at,
        )
        append_rows_to_csv(self.csv_path, rows)

        # 5. Save the per-image metadata.
        save_json(
            image_dir / "metadata.json",
            {
                "setup_id": self.setup_id,
                "image_id": image_id,
                "created_at": created_at,
                "valid_game_position": self.valid_game_position,
                "legal_move_mode": self.legal_mode,
                "board_fen": board_fen,
                "previous_board_fen": self.previous_board_fen,
                "move_uci": self.move_uci,
                "piece_map": piece_map,
                "full_image_path": str(full_image_path),
                "warped_image_path": str(warped_image_path),
                "squares_dir": str(squares_dir),
                "calibration_metadata_path": str(self.calibration_metadata_path),
            },
        )

        self.image_count += 1
        print(f"Captured {image_id}: appended {len(rows)} rows to {self.csv_path}")
        return True


# ---------------------------------------------------------------------------
# Pygame GUI (lazily imports pygame so the helpers above stay import-light)
# ---------------------------------------------------------------------------


class BoardUI:
    """A deliberately simple pygame board for driving the capture session."""

    SQUARE = 80
    MARGIN = 40
    PANEL = 320

    LIGHT = (240, 217, 181)
    DARK = (181, 136, 99)
    SELECT = (246, 246, 105)
    BG = (30, 30, 34)
    TEXT = (230, 230, 230)

    def __init__(self, session: DataGenerationSession) -> None:
        import pygame  # local import: not needed for the helpers/tests

        self.pg = pygame
        self.session = session

        pygame.init()
        pygame.display.set_caption("Chess data generation")
        board_px = self.SQUARE * 8
        self.width = self.MARGIN * 2 + board_px + self.PANEL
        self.height = self.MARGIN * 2 + board_px
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.piece_font = pygame.font.SysFont("consolas", 46, bold=True)
        self.label_font = pygame.font.SysFont("consolas", 16)
        self.panel_font = pygame.font.SysFont("consolas", 18)

        # Interaction state.
        self.selected: str | None = None          # source square for a move
        self.spawn: str | None = None              # piece selected to place
        self.promotion: tuple[str, str] | None = None  # (from, to) awaiting choice
        self.message = "Press 'r' to calibrate a setup, then Space to capture."
        self.running = True

    # -- coordinate helpers -------------------------------------------------

    def _square_rect(self, square: str):
        file_idx = ord(square[0]) - ord("a")
        rank = int(square[1])
        col = file_idx
        row = 8 - rank  # White at the bottom
        x = self.MARGIN + col * self.SQUARE
        y = self.MARGIN + row * self.SQUARE
        return self.pg.Rect(x, y, self.SQUARE, self.SQUARE)

    def _square_at_pixel(self, pos) -> str | None:
        x, y = pos
        col = (x - self.MARGIN) // self.SQUARE
        row = (y - self.MARGIN) // self.SQUARE
        if 0 <= col < 8 and 0 <= row < 8:
            file_char = chr(ord("a") + int(col))
            rank = 8 - int(row)
            return f"{file_char}{rank}"
        return None

    # -- drawing ------------------------------------------------------------

    def _draw_piece(self, symbol: str, rect) -> None:
        is_white = symbol.isupper()
        fill = (245, 245, 245) if is_white else (25, 25, 25)
        outline = (25, 25, 25) if is_white else (245, 245, 245)
        base = self.piece_font.render(symbol, True, fill)
        line = self.piece_font.render(symbol, True, outline)
        centered = base.get_rect(center=rect.center)
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            self.screen.blit(line, (centered.x + dx, centered.y + dy))
        self.screen.blit(base, centered)

    def _draw_board(self) -> None:
        for square in SQUARES:
            rect = self._square_rect(square)
            file_idx = ord(square[0]) - ord("a")
            rank = int(square[1])
            is_light = (file_idx + rank) % 2 == 0
            self.pg.draw.rect(self.screen, self.LIGHT if is_light else self.DARK, rect)
            if square == self.selected:
                self.pg.draw.rect(self.screen, self.SELECT, rect, 5)
            if self.promotion and square in self.promotion:
                self.pg.draw.rect(self.screen, (200, 60, 60), rect, 5)

            piece = self.session.board.piece_at(chess.parse_square(square))
            if piece is not None:
                self._draw_piece(piece.symbol(), rect)

    def _draw_panel(self) -> None:
        x = self.MARGIN * 2 + self.SQUARE * 8
        mode = "LEGAL" if self.session.legal_mode else "FREE-PLACEMENT"
        lines = [
            f"Setup: {self.session.setup_id or '(none)'}",
            f"Images captured: {self.session.image_count}",
            f"Mode: {mode}",
            f"valid_game_position: {self.session.valid_game_position}",
            f"Last move: {self.session.move_uci or '-'}",
            f"Spawn piece: {self.spawn or '-'}",
            "",
        ]
        if self.promotion:
            lines.append("PROMOTION: press q / r / b / n")
            lines.append("")
        lines += [
            "-- Controls --",
            "Space: capture",
            "r: recalibrate / new setup",
            "n: new game (legal mode)",
            "f: toggle legal/free mode",
            "click: select then move",
            "right-click: cancel",
            "",
            "Free mode:",
            "  P N B R Q K = white spawn",
            "  p n b r q k = black spawn",
            "  x: remove under cursor",
            "",
            "Esc: quit  (q also quits",
            "     in legal mode)",
            "",
            f"> {self.message}",
        ]
        y = self.MARGIN
        for line in lines:
            surface = self.panel_font.render(line, True, self.TEXT)
            self.screen.blit(surface, (x, y))
            y += 22

    def _draw(self) -> None:
        self.screen.fill(self.BG)
        self._draw_board()
        self._draw_panel()
        self.pg.display.flip()

    # -- event handling -----------------------------------------------------

    def _on_key(self, event) -> None:
        pg = self.pg
        key = event.key
        char = event.unicode

        if key == pg.K_ESCAPE:
            if self.promotion:
                self.promotion = None
                self.message = "Promotion cancelled."
            else:
                self.running = False
            return

        if key == pg.K_SPACE:
            self._do_capture()
            return

        # Promotion choice takes priority while pending.
        if self.promotion:
            if char in PROMOTION_MAP:
                self._finish_promotion(char)
            else:
                self.message = "Choose promotion: q / r / b / n (Esc cancels)."
            return

        if char == "f":
            self.session.toggle_mode()
            self.selected = None
            self.spawn = None
            self.message = f"Mode: {'LEGAL' if self.session.legal_mode else 'FREE'}"
            return

        if self.session.legal_mode:
            # In legal mode the letters are commands, not piece spawns.
            if char == "n":
                self.session.new_game()
                self.selected = None
                self.message = "New game."
            elif char == "r":
                self._do_recalibrate()
            elif char == "q":
                self.running = False
            return

        # Free-placement mode: letters spawn pieces, 'x' removes.
        if char == "x":
            square = self._square_at_pixel(pg.mouse.get_pos())
            if square:
                self.session.remove_piece(square)
                self.message = f"Removed piece on {square}."
            else:
                self.message = "Hover a square, then press x to remove."
        elif char in PIECE_SYMBOLS:
            self.spawn = char
            self.selected = None
            self.message = f"Spawn '{char}': click a square to place it."

    def _on_click(self, event) -> None:
        if event.button == 3:  # right-click cancels current selection
            self.selected = None
            self.spawn = None
            self.message = "Selection cancelled."
            return
        if event.button != 1:
            return

        square = self._square_at_pixel(event.pos)
        if square is None:
            return

        # Free-placement: if a spawn piece is armed, place it.
        if not self.session.legal_mode and self.spawn:
            self.session.place_piece(square, self.spawn)
            self.message = f"Placed '{self.spawn}' on {square}."
            return

        # Otherwise: click source, then target.
        if self.selected is None:
            self.selected = square
            self.message = f"Selected {square}."
            return

        source = self.selected
        self.selected = None
        if source == square:
            self.message = "Deselected."
            return

        if self.session.legal_mode:
            result = self.session.apply_legal_move(source, square)
            if result == "ok":
                self.message = f"Move {self.session.move_uci}."
            elif result == "need_promotion":
                self.promotion = (source, square)
                self.message = "Promotion: press q / r / b / n."
            else:
                self.message = f"Illegal move {source}{square}."
        else:
            if self.session.move_piece_free(source, square):
                self.message = f"Moved {source} -> {square}."
            else:
                self.message = f"No piece on {source}."

    def _finish_promotion(self, char: str) -> None:
        source, target = self.promotion
        self.promotion = None
        result = self.session.apply_legal_move(
            source, target, promotion=PROMOTION_MAP[char]
        )
        if result == "ok":
            self.message = f"Promoted: {self.session.move_uci}."
        else:
            self.message = "Promotion failed."

    def _do_capture(self) -> None:
        self.message = "Capturing..."
        self._draw()
        ok = self.session.capture()
        self.message = "Captured." if ok else "Capture failed (see console)."

    def _do_recalibrate(self) -> None:
        self.message = "Calibrating in the OpenCV window..."
        self._draw()
        ok = self.session.start_new_setup()
        if ok:
            self.selected = None
            self.spawn = None
            self.promotion = None
            self.message = f"New setup {self.session.setup_id}."
        else:
            self.message = "Setup failed (see console). Press 'r' to retry."

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        pg = self.pg
        clock = pg.time.Clock()
        # Try to establish an initial setup up front.
        if self.session.processor is None:
            self._draw()
            self.session.start_new_setup()

        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.KEYDOWN:
                    self._on_key(event)
                elif event.type == pg.MOUSEBUTTONDOWN:
                    self._on_click(event)
            self._draw()
            clock.tick(30)

        pg.quit()


def generate_data(
    config_path: Path = Path("config.yaml"),
    data_root: Path = DATA_ROOT,
) -> None:
    """Entry point: open the virtual board and run the capture loop."""
    session = DataGenerationSession(config_path=Path(config_path), data_root=Path(data_root))
    ui = BoardUI(session)
    ui.run()


if __name__ == "__main__":
    generate_data()



"""
What I built

I replaced the broken skeleton in src/chess_assistant/data_generation.py (it had syntax errors and undefined functions like piece_at) with a working data-generation tool, plus tests and a dependency update.

How the existing code behaves (and how I reused it)

- calibrate(setup_dir) — blocking OpenCV flow; on success writes setup_dir/calibration_metadata.json, returns the dict (or None). → called from start_new_setup().
- capture_image(setup_dir) — creates its own board_<timestamp>/ subdir with image.png and returns that dir. → I use its folder name as image_id, so I don't invent an incompatible convention.
- Processor.warp(img) → <dir>/image_warped.png; Processor.cutout(warped) → <dir>/squares/<sq>/<sq>.png (+ _annotated.png, _masked.npy, _metadata.json). → reused as-is; I only append a "label" key to each existing <sq>_metadata.json.
- SQUARES drives all 64-square iteration; labels use python-chess piece symbols.

How to run

uv run python -m chess_assistant.data_generation
(Requires the robot + a display. It auto-starts calibration on launch.)

Keybindings

┌──────────────────────────────────────┬──────────────────────────────────────────────────────────────────────┐
│                 Key                  │                                Action                                │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Space                                │ Capture: photo → warp → cutout → labels → 64 CSV rows                │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ r                                    │ Recalibrate / new setup                                              │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ n                                    │ New game (legal mode only)                                           │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ f                                    │ Toggle legal ↔ free-placement                                        │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ click → click                        │ Select source, then target (move)                                    │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ right-click                          │ Cancel selection / armed spawn                                       │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ q / Esc                              │ Quit (in legal mode)                                                 │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ free mode: P N B R Q K / p n b r q k │ Arm white/black piece to spawn (then click)                          │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ free mode: x                         │ Remove piece under cursor                                            │
├──────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ promotion: q r b n                   │ Choose promotion piece (never silent-defaults to queen; Esc cancels) │
└──────────────────────────────────────┴──────────────────────────────────────────────────────────────────────┘

Data layout

data/generated/
  data.csv
  <setup_id>/                      # timestamp
    calibration_metadata.json      # from calibrate()
    setup_metadata.json
    board_<timestamp>/             # = image_id, from capture_image()
      image.png
      image_warped.png
      metadata.json
      squares/<sq>/<sq>.png, <sq>_annotated.png, <sq>_masked.npy, <sq>_metadata.json(+label)

CSV schema (one row per square)

setup_id, image_id, square, label, square_image_path, full_image_path, calibration_metadata_path, valid_game_position, board_fen, previous_board_fen, move_uci, created_at — stdlib csv append mode, header written once.

Labels

13-class from the virtual board: "empty" or P N B R Q K / p n b r q k. valid_game_position is True only while the game has stayed in legal mode since the last new game/setup; any free-placement edit flips it to False.

Verification

- Added tests/test_data_generation.py — 13 tests pass: labels, piece_map, nested paths, row shape, CSV header-once/append, create_setup, legal-move FEN tracking, illegal rejection, promotion-requires-choice, free-mode invalidation, remove/spawn/move, new-game reset.
- Headless smoke check confirmed BoardUI constructs, draws, and orients White-at-bottom.
- Added pygame>=2.6.0 to deps and a dev group with pytest; ran uv sync.

Assumptions & limitations

- image_id = the board_<timestamp> folder capture_image creates; setups are timestamp-named (no silent overwrites).
- pygame is imported lazily inside BoardUI, so helpers/tests need no display.
- In free mode, n/r don't trigger new-game/recalibrate (those letters spawn black knight/rook); press f to return to legal mode first — this avoids key collisions. Quit from free mode with Esc.
- Free-mode board_fen is best-effort (None if python-chess can't represent it); the piece_map in each metadata.json is always the reliable label source.
- GUI/robot/camera paths aren't unit-tested (no hardware/display here), per your choice.

One thing to flag: I set the data root to data/generated/ (per your spec), which differs from the old skeleton's data/ — let me know if you'd rather keep the flat data/ root.
"""