import chess

from chess_assistant.config import SQUARES, PIECES

class ChessGame:
    """
    Requires a method that takes as input a board prediction, and based on that
    prediction, evaluates every possible legal move for plausibility.
    Should then output the most likely move.
    Perhaps also keep track of multiple moves. 
    (Easier to debug: if it's not a specific move that I think it is, can try a new one.)
    """
    def __init__(self, fen: str | None = None, model_type: str = "LLM"):
        self.board = chess.Board(fen=fen) 
            # allow for passing of FEN to simulate continuation of game from an 
            # arbitrary prior position during evaluation
        assert model_type in ["LLM", "CNN"]
        self.model_type = model_type
    
    def fen(self):
        return self.board.fen()

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
            for piece in PIECES:
                piece_at_square = self.board.piece_at(chess.parse_square(square)).symbol()
                if (
                    (piece == "empty" and piece_at_square is None)
                    or piece == piece_at_square
                ):
                    initial_loss += (1 - getattr(board_estimate, f"{square}.{piece}")) ** 2
                else:
                    initial_loss += getattr(board_estimate, f"{square}.{piece}") ** 2
            
            # QUESTION: do I also want to take predictions for other pieces into account?
            # E.g. 3 pieces. if i have 
                # A: 0, 0,8, 0.2
                # B: 0.4, 0.4, 0.2
            # This second way (real MSE?) the score for the third piece would be lower with B.
            # Perhaps check out some common classification losses here.
            # Then access the prediction in board_estimate for that piece
            # And add squared deviation to initial_score

        scored_moves = {}
        for move in self.board.legal_moves:
            loss_increment = 0

            before = self.board.copy()
            after = self.board.copy()
            after.push(move)

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
                
                So for new piece, we add (1 - x) ** 2 - x ** 2 = -2x + 1
                For old piece, we subtract -2x + 1
                """
                loss_increment += -2 * getattr(board_estimate, f"{square}.{new_piece}") + 1
                loss_increment += 2 * getattr(board_estimate, f"{square}.{old_piece}") - 1
            
            scored_moves[move.uci()] = loss_increment
        
        # Sort candiate moves by likelihood (descending), which means
        # sort in ascending order by error impact of candidate moves
        return [move for move in sorted(scored_moves.items(), key=lambda x: x[1])]

    def apply_move(self, move_uci):
        move = chess.Move.from_uci(move_uci)

        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move_uci}")
    
        self.board.push(move)

    def print_board(self):
        print(self.board)


if __name__ == "__main__":
    game = ChessGame()
    print(game.board)
    for move in game.board.legal_moves: 
        print(move)
    print(type(next(iter(game.board.legal_moves)).uci()))
    breakpoint()