import threading

import chess
import chess.engine
import torch
from torch import nn

from chess_assistant.config import SQUARES, PIECES
from chess_assistant.model.config import TARGET_MAP, INVERSE_TARGET_MAP

STOCKFISH_PATH = r"C:\Users\User\AppData\Local\Microsoft\WinGet\Packages\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe\stockfish\stockfish-windows-x86-64-avx2.exe"

class ChessGame:
    """
    Requires a method that takes as input a board prediction, and based on that
    prediction, evaluates every possible legal move for plausibility.
    Should then output the most likely move.
    Perhaps also keep track of multiple moves. 
    (Easier to debug: if it's not a specific move that I think it is, can try a new one.)
    """
    def __init__(self, fen: str | None = None, model_type: str = "LLM", depth=16):
        fen = fen if fen is not None else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.board = chess.Board(fen=fen) 
            # allow for passing of FEN to simulate continuation of game from an 
            # arbitrary prior position during evaluation
        assert model_type in ["LLM", "CNN"]
        self.model_type = model_type
        if self.model_type == "CNN":
            self.loss_fn = nn.CrossEntropyLoss()
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        self.depth = depth
            # maybe increase to 18 eventually
        # Speaker's background thread calls cp_loss_for() while the main thread may still be
        # inside estimate_move(). One Stockfish process, so serialise access to it.
        self.engine_lock = threading.Lock()
        self.recent_position_score = self.eval_position()

        # Game history, appended to by apply_move(). Feeds the commentary prompt.
        self.move_log = []  # one dict per played move: uci, san, turn, capture, cp_loss
        self.cp_losses = {"white": [], "black": []}

    def fen(self):
        return self.board.fen()

    def eval_position(self, board=None):
        board = self.board if board is None else board
        with self.engine_lock:
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score = info["score"].white().score(mate_score=10000)
        return score  # centipawns, from White's perspective

    def estimate_move(self, board_estimate):
        # board_estimate object has 64 fields (a1, ...)
        # each of those fields is a square estimate object:
            # which has some metadata fields
            # and then also K, Q, ... along with floats for confidence.
        # This should allow me to access the predictions I need.
        # Score: Mean Absolute Error? Or Mean Squared Error?
        # I think mean squared error makes sense. 
        # Task becomes: for each move, calculate Mean Squared error, then sort by MSE.

        """
        Note that this initial score does not even matter! 
        This is just a constant term we add to every move score, which therefore
        does not impact the ordering of candidate moves.
        """
        # Score based on the previous board position.
        initial_loss = 0
        for square in SQUARES:
            square_estimate = getattr(board_estimate, square)
            if self.model_type == "LLM":
                for piece in PIECES:
                    # TODO: this is not robust to piece_
                    piece_at_square = self.board.piece_at(chess.parse_square(square))
                    piece_at_square = "empty" if piece_at_square is None else piece_at_square.symbol()
                    if (
                        (piece == "empty" and piece_at_square is None)
                        or piece == piece_at_square
                    ):
                        initial_loss += (1 - getattr(square_estimate, piece)) ** 2
                    else:
                        initial_loss += getattr(square_estimate, piece) ** 2
            else: # use cross-entropy loss
                piece_at_square = self.board.piece_at(chess.parse_square(square))
                piece_at_square = "empty" if piece_at_square is None else piece_at_square.symbol()
                initial_loss += self.loss_fn(
                    torch.tensor(
                        [
                            getattr(square_estimate, INVERSE_TARGET_MAP[target])
                            for target in range(13) 
                            # the double for loop here is a bit convoluted;
                            # reason: each target maps to exactly one piece; therefore
                            # no need for iteration
                        ]
                        , 
                        dtype=torch.float32
                    ),    
                    torch.tensor(TARGET_MAP[piece_at_square])
                )

            # QUESTION: do I also want to take predictions for other pieces into account?
            # E.g. 3 pieces. if i have 
                # A: 0, 0,8, 0.2
                # B: 0.4, 0.4, 0.2
            # This second way (real MSE?) the score for the third piece would be lower with B.
            # Perhaps check out some common classification losses here.
            # Then access the prediction in board_estimate for that piece
            # And add squared deviation to initial_score

        scored_moves = []
        for move in self.board.legal_moves:
            # Score the move
            loss_increment = 0

            before = self.board.copy()
            after = self.board.copy()
            after.push(move)

            move_info = self.describe_move(move, after=after)

            changed_squares = []

            """
            Iterate through all squares rather than just the squares we can read out from the 
            move in UCI notation, because there are edge cases in which >2 squares get impacted:
                - Castling changes four squares: king from/to and rook from/to.
                - En passant changes three squares: pawn from, pawn to, and captured pawn square.
                - Promotion capture still works for the destination square, 
                  but the captured piece disappears from square_to.
            
            # TODO: Could slightly increase speed of this by providing manual code for these edge cases
            -> we don't always loop through all 64 squares for every move.
            """
            for square in chess.SQUARES:
                before_piece = before.piece_at(square)
                after_piece = after.piece_at(square)

                before_symbol = before_piece.symbol() if before_piece else "empty"
                after_symbol = after_piece.symbol() if after_piece else "empty"

                if before_symbol != after_symbol:
                    changed_squares.append((
                        chess.square_name(square),
                        before_symbol,
                        after_symbol,
                    ))

            for square, old_piece, new_piece in changed_squares:
                """
                For a given square:
                1) remove the earlier contribution from old piece and new piece,
                2) add the new contribution from old piece and new piece.
                
                score += (1 - getattr(board_estimate, f"{square}.{new_piece}")) ** 2
                score -= (1 - getattr(board_estimate, f"{square}.{old_piece}")) ** 2

                score -= getattr(board_estimate, f"{square}.{new_piece}") ** 2
                score += getattr(board_estimate, f"{square}.{old_piece}") ** 2

                CAREFUL: getattr does apparently not support nested attribute access.
                
                So for new piece, we add (1 - x) ** 2 - x ** 2 = -2x + 1
                For old piece, we subtract -2x + 1
                """
                square_estimate = getattr(board_estimate, square)
                if self.model_type == "LLM":
                    loss_increment += -2 * getattr(square_estimate, new_piece) + 1
                    loss_increment += 2 * getattr(square_estimate, old_piece) - 1
                else:
                    square_pred_tensor = torch.tensor(
                        [
                            getattr(square_estimate, INVERSE_TARGET_MAP[target])
                            for target in range(13)
                        ],
                        dtype=torch.float32
                    ) 
                    loss_increment += self.loss_fn(square_pred_tensor, torch.tensor(TARGET_MAP[new_piece]))
                    loss_increment -= self.loss_fn(square_pred_tensor, torch.tensor(TARGET_MAP[old_piece]))

            scored_moves.append({
                "move": move.uci(),
                "loss": initial_loss + loss_increment,
                "move_info": move_info,
            })
            # Computation of initial loss is not necessary; it does not affect ranking.
            # Keeping it as an evaluation metric for now.

        # Sort candiate moves by likelihood (descending), which means
        # sort in ascending order by error impact of candidate moves
        scored_moves.sort(key = lambda x: x["loss"])
        return scored_moves

    def describe_move(self, move, after=None):
        """Everything a commentator would want to know about `move`, read off the
        *pre-move* board. Nothing here mutates state, so a candidate move can be
        described without ever being played.

        `after` is the board with `move` already pushed. estimate_move() builds one
        anyway, so it passes it in; otherwise we make our own.
        """
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        board = self.board
        if after is None:
            after = board.copy()
            after.push(move)

        piece = board.piece_at(move.from_square)
        assert piece is not None  # otherwise the move could not have been legal

        en_passant = board.is_en_passant(move)
        if en_passant:
            # The captured pawn does not stand on the destination square, so
            # piece_at(to_square) is None here. It is always a pawn.
            captured_piece = "Pawn"
        else:
            captured = board.piece_at(move.to_square)
            captured_piece = _piece_name(captured.piece_type) if captured else None

        if board.is_castling(move):
            castle = "kingside" if board.is_kingside_castling(move) else "queenside"
        else:
            castle = None

        return {
            "move": move.uci(),
            "san": board.san(move),
            "moved_piece": _piece_name(piece.piece_type),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "move_number": board.fullmove_number,
            "capture": board.is_capture(move),
            "captured_piece": captured_piece,
            "castle": castle,
            "en_passant": en_passant,
            "promotion": _piece_name(move.promotion) if move.promotion else None,
            "check": board.gives_check(move),
            "checkmate": after.is_checkmate(),
        }

    def evaluate_after(self, move):
        """White-perspective centipawn score of the position `move` would lead to.
        Does not touch self.board, so it is safe to call from a worker thread on a
        move that has not been (and may never be) played."""
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        board = self.board.copy()
        board.push(move)
        return self.eval_position(board)

    def cp_loss_for(self, move):
        """How much `move` costs the side playing it, in centipawns. Returns
        (cp_loss, new_score) so the caller can reuse the evaluation when it commits
        the move instead of paying for a second engine analysis.

        The mover's colour is read off the board *before* any push. Doing this after
        pushing would read the opponent's colour and invert the sign, which is what
        the old rate_move() did.
        """
        new_score = self.evaluate_after(move)
        if self.board.turn == chess.WHITE:
            cp_loss = self.recent_position_score - new_score
        else:
            cp_loss = new_score - self.recent_position_score
        # A negative loss would mean the move beat the engine's own best line -> clamp.
        return max(0, cp_loss), new_score

    def apply_move(self, move_uci, move_info=None, cp_loss=None, new_score=None):
        move = chess.Move.from_uci(move_uci)

        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move_uci}")

        if move_info is None:
            move_info = self.describe_move(move)
        if cp_loss is None or new_score is None:
            cp_loss, new_score = self.cp_loss_for(move)

        self.board.push(move)
        self.recent_position_score = new_score

        self.move_log.append({
            "uci": move_info["move"],
            "san": move_info["san"],
            "turn": move_info["turn"],
            "capture": move_info["capture"],
            "cp_loss": cp_loss,
        })
        self.cp_losses[move_info["turn"]].append(cp_loss)

    # --- history, as consumed by the commentary prompt ---

    def recent_moves(self, n=6):
        return [entry["uci"] for entry in self.move_log[-n:]]

    def capture_streak(self):
        """How many moves in a row, ending at the last one played, were captures."""
        return _trailing_streak(self.move_log, capture=True)

    def quiet_streak(self):
        """How many moves in a row, ending at the last one played, were not captures."""
        return _trailing_streak(self.move_log, capture=False)

    def average_cp_loss(self):
        return {
            side: (sum(losses) / len(losses)) if losses else 0.0
            for side, losses in self.cp_losses.items()
        }

    def last_cp_losses(self, n=5):
        return {side: losses[-n:] for side, losses in self.cp_losses.items()}

    def history_snapshot(self, recent_moves=6, recent_cp_losses=5):
        """A self-contained copy of the game history, safe to hand to a worker thread
        (the main thread keeps mutating move_log/cp_losses as the game goes on)."""
        return {
            "recent_moves": self.recent_moves(recent_moves),
            "capture_streak": self.capture_streak(),
            "quiet_streak": self.quiet_streak(),
            "average_cp_loss": self.average_cp_loss(),
            "last_cp_losses": self.last_cp_losses(recent_cp_losses),
        }

    def print_board(self):
        print(self.board)

    def close(self):
        self.engine.quit()


def _piece_name(piece_type):
    return chess.piece_name(piece_type).capitalize()


def _trailing_streak(move_log, capture):
    streak = 0
    for entry in reversed(move_log):
        if entry["capture"] != capture:
            break
        streak += 1
    return streak


if __name__ == "__main__":
    game = ChessGame()
    print(game.board)
    for move in game.board.legal_moves: 
        print(move)
    print(type(next(iter(game.board.legal_moves)).uci()))
    breakpoint()