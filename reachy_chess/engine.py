import chess
import chess.engine

class ChessEngine:
    def __init__(self, stockfish_path=r"C:\Users\User\AppData\Local\Microsoft\WinGet\Packages\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe\stockfish\stockfish-windows-x86-64-avx2.exe"):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    def get_best_move(self, fen, time_limit=0.5):
        board = chess.Board(fen)
        result = self.engine.play(
            board,
            chess.engine.Limit(time=time_limit),
        )
        return result.move.uci()
    
    def close(self):
        self.engine.quit()