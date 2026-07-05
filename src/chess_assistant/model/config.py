TARGET_MAP = {piece: label for label, piece in enumerate([
    "empty", 
    "K", "Q", "R", "B", "N", "P",
    "k", "q", "r", "b", "n", "p"
])}

INVERSE_TARGET_MAP = {label: piece for piece, label in TARGET_MAP.values()}