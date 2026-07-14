"""
Board vocabulary shared across the project: the 64 square names and the 13 square labels
(12 pieces in FEN notation, plus "empty").
"""
FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS = [str(i) for i in range(1, 9)]
SQUARES = [file + rank for file in FILES for rank in RANKS]

PIECES = ["empty", "K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]