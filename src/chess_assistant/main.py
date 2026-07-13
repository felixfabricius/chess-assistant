from pathlib import Path
from omegaconf import OmegaConf

from chess_assistant.setup import setup
from chess_assistant.camera import capture_image
from chess_assistant.vision import BoardEstimator
from chess_assistant.game import ChessGame
from chess_assistant.image_processing import Processor
from chess_assistant.input import InputDetector
from chess_assistant.robot import Speaker
from chess_assistant.calibration import make_head_rigid, move_to_capture_pose
from chess_assistant.calibration_monitor import launch_calibration_monitor

import json
import time
from datetime import datetime
from reachy_mini import ReachyMini


def main(mini) -> None:
    
    config = OmegaConf.load("config.yaml")

    setup_dir, pixel_coordinates, robot_pose = setup(mini)
    calibration_metadata_path = setup_dir / "calibration_metadata.json"
    # Live drift check in its own process: overlays the calibrated corners on the undistorted
    # camera feed so you can watch whether the camera has moved away from the calibrated pose.
    launch_calibration_monitor(calibration_metadata_path)
    image_processor = Processor(calibration_metadata_path, "config.yaml")
    board_estimator = BoardEstimator("CNN", config, calibration_metadata_path, model_path=config.vision.model_path, device="cpu")
    input_detector = (
        InputDetector(input_type="robot", mini=mini, calibration_metadata_path=calibration_metadata_path) 
        if config.get("input", {"source": "robot"}).get("source", "robot") == "robot" 
        else InputDetector(input_type="keyboard", target_key=None)
    )
    speaker = Speaker(mini)
    game = ChessGame()

    # Keep the head in a rigid, non-drifting hold for the whole game so every board image is
    # captured from the calibrated pose (see make_head_rigid). Re-asserted here in case setup
    # loaded an existing calibration without running calibrate().
    make_head_rigid(mini)

    print("Start Game")
    # Build game loop
    game_over = False
    while not game_over:
        print("Ready for move")
        move_made = input_detector.detect_input(type="move_made")
        print(f"move made: {move_made}") # this should definitely be true
        # Snap the head back to the exact calibrated pose so every board image is taken from
        # the same position (the head may have drifted while operating the antennas).
        move_to_capture_pose(mini, *robot_pose)
        image_dir = capture_image(mini, setup_dir)
        print("image captured")
        # Use antennas to get camera input
        # Perhaps we can use camera orientation to determine on which side the white player is
        # sitting; and then the AntennasInput class can keep track of everyhing else.
        # Including on which side the next input needs to occur.

        # Capture image from camera
        warped_image_path = image_processor.warp(image_dir / "image.png")
        squares_dir = image_processor.cutout(warped_image_path)
        board_estimate = board_estimator.estimate_board(squares_dir)
        move_estimates = game.estimate_move(board_estimate)
        print("moves estimated")


        move_estimate_accepted = False
        for move in move_estimates:
            speaker.suggest_move(move["move"])
            move_estimate_rejected = input_detector.detect_input(type="move_estimate_rejected", time=config.get("review_time", 3))
            move_estimate_accepted = not move_estimate_rejected:
            if move_estimate_accepted:
                break
        
        move = move["move"]
        assert move_estimate_accepted # Assert we didn't loop through all moves without accepting any.
        print(f"move estimate accepted: {move}")
        moved_piece = game.identify_moved_piece(move)
        game.apply_move(move)
        move_cp_loss = game.rate_move(move)

        speaker.comment_on_move(move, move_cp_loss, moved_piece)

        game_over = game.board.is_checkmate()
        if game_over:
            speaker.exclaim_win(game)

        input_detector.switch_turn()

if __name__ == "__main__":
    with ReachyMini(media_backend="default") as mini:
        main(mini)
