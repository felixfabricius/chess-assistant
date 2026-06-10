import chess

class ChessGame:
    def __init__(self):
        self.board = chess.Board()
    
    def fen(self):
        return self.board.fen()

    def apply_move(self, move_uci):
        move = chess.Move.from_uci(move_uci)

        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move_uci}")
    
        self.board.push(move)

    def print_board(self):
        print(self.board)