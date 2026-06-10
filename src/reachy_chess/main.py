from reachy_chess.game import ChessGame
from reachy_chess.engine import ChessEngine
#from reachy_chess.robot import RobotOutput
#from reachy_chess.vision import ManualInput

breakpoint()
print("Hello world")
"""
def main():
    game = ChessGame()
    engine = ChessEngine()
    robot = RobotOutput()
    vision = ManualInput()

    try:
        while True:
            game.print_board()

            user_move = vision.get_move()

            if user_move == "quit":
                break

            try:
                game.apply_move(user_move)
            except ValueError as e:
                print(e)
                continue

            if game.board.is_game_over():
                print("Game over.")
                break

            best_move = engine.get_best_move(game.fen())
            robot.say_best_move(best_move)
            game.apply(best_move)

            if game.board.is_game_over():
                print("Game over.")
                break

    finally:
        engine.close()

if __name__ == "__main__":
    main()
"""