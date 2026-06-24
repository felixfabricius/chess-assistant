from pathlib import Path
import cv2
import numpy as np
from omegaconf import OmegaConf
import json

class Processor:
    def __init__(self, metadata_path: Path, config_path: Path | None = None) -> None:
        """
        Store attributes:
        - how many pixels to allocate for padding in each direction
        - size of output image (accounting for padding)
        - matrix to use for image warping (accounting for padding)
        """
        # Get board size of transformed image (excl. padding) from config
        board_size = None
        if config_path:
            config = OmegaConf.load(config_path)
            board_size = config.get("image_processing", OmegaConf.create()).get("size")
        board_size = board_size if board_size else 400
        last_coordinate = board_size - 1

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # TODO
        # Order of ordered_corners
        # Dict which stores e.g. {"tl": "a8"}
        self.corner_map = metadata["camera_natural_orientation"]["order"]

        src = np.array(
            [
                metadata["actual_corners_px"][corner_square] 
                for corner_square in 
                [self.corner_map[square_position] for square_position in ["tl", "tr", "br", "bl"]]
            ],
            dtype=np.float32
        )

        dst_initial = np.array(
            [
                [0, 0], # top-left
                [last_coordinate, 0], # top_right
                [last_coordinate, last_coordinate],
                [0, last_coordinate]
            ],
            dtype=np.float32
        )

        matrix_initial = cv2.getPerspectiveTransform(src, dst_initial)

        # Use initial matrix to transform the extended corners and calculate padding based on that
        # At this point we have:
            # metadata["extended_corners_px"]["a8"]
            # and self.corner_map also taken from metadata["corner_map"], which e.g. does "tl": "a8"
        # Calculate pixel position of extended pixels.
        src_extended_corners = np.array(
            [
                metadata["extended_corners_px"][corner_square]
                for corner_square in
                [self.corner_map[square_position] for square_position in ["tl", "tr", "br", "bl"]]
            ],
            dtype=np.float32
        ) # Shape: (4, 2)

        dst_extended_corners_initial = (
            cv2.perspectiveTransform(
                src_extended_corners.reshape(4, 1, 2), # this is expected format
                matrix_initial
            ) # this yields array with shape (4, 1, 2)
            .reshape(4, 2)
        ) # this is now in familiar format and can be compared with the dst array

        # Get padding
        # Recall that order corresponds to tl, tr, br, bl
        pixel_differences = dst_extended_corners_initial - dst_initial 
            # Subtraction will be item-by-item
            # Interpreting result:
            # Subtracting the pixels yields 4 vectors
            # First element of each of those 4 vectors will be x-difference.
                # If > 0, then extended corner is projected further to the right
                  # e.g. for the tr corner, with my current setup, would expect to get
                  # first element > 0 
                # If < 0: then extended corner is projected further to the left
            # Second element:
                # y-difference. If > 0, then extended corner is further below -> need to pad downwards
                # if < 0, extended corner is further above
            # Can extract padding by taking the maximum of these dimensions
        # p_up: 'padding_up'. How many pixels to add to top? of square
        p_up = round(1.05 * max(np.max(-pixel_differences, axis=0)[1], 0))
            # Take maximum of negative difference 
            # (negative means extended corner is above actual corner)
            # across the 4 corners.
            # Also floor padding in every direction at 0.
        p_down = round(1.05 * max(np.max(pixel_differences, axis=0)[1], 0))

        p_left = round(1.05 * max(np.max(-pixel_differences, axis=0)[0], 0))
        p_right = round(1.05 * max(np.max(pixel_differences, axis=0)[0], 0))

        padding = {
            "up": p_up,
            "down": p_down,
            "left": p_left,
            "right": p_right
        }

        # Given self.padding, modify size of image and create new matrix
        size_x = board_size + padding["left"] + padding["right"]
        size_y = board_size + padding["up"] + padding["down"]

        # Modify destination coordinates of the board corners to account for padding
        dst = np.array(
            [
                [padding["left"], padding["up"]], # top-left
                [padding["left"] + board_size, padding["up"]], # top-right
                [padding["left"] + board_size, padding["up"] + board_size], # bottom-right
                [padding["left"], padding["up"] + board_size]
            ],
            dtype=np.float32
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        
        # Store attributes
        self.padding = padding
        self.board_size = board_size
        self.image_size = (size_x, size_y)
        self.matrix = matrix

    def warp(self, image_path: Path) -> Path:
        image = cv2.imread(image_path)
        warped_image = cv2.warpPerspective(image, self.matrix, self.image_size)
        warped_image_path = image_path.parent / (str(image_path.stem) + "_warped.png")
        cv2.imwrite(str(warped_image_path), warped_image)
        # Note that in this case there is no need to specify some color map transformation
        # That is only needed, when passing the image to something that expects RGB
        return warped_image_path
    
    def cutout(self, warped_image_path):
        """
        Use self.corner map to accurately label squares
        That maps e.g. "tl": "a8"
        
        Question: how to adequately determine pixel position of a given square, 
        given 
        - self.padding
        - self.size
        - and self.corner_map

        If we have tl: a8, then it's quite easy.
        Just find position of bottom left corner.
        Then given bottom left corner, use self.padding and self.size 
        to get adequate corners for the crop.

        This approach to finding the pixel positions of a square also works if another 
        square is in the top left. Yes. 

        Question is just: how to label, given e.g. the pixel coordinate of the bottom left
        corner of a square. (Here bottom left means "bottom left in the image").
        Let's also say we know the square increment.

        Let's do the loop such that we change the pixel coordinate of the bottom 
        left corner of square.
        How: subtract 1, 2, ..., 8 times square height from the y-coordinate (i index)
        And add: 0, 1, 2, ..., 7 times the square width (= square height) to the x-coordinate (j index)

        How to use this for square labelling?
        If top left = a8:
            - Vertical:
              i = 1 -> 8; i = 2 -> 7 etc.
            - Horizontal:
              j = 0 -> a, j = 1 -> b, ...
        
        Elif top left = a1:
            - Vertical:
              i = 1 -> a, i = 2 -> b, i = 3 -> c, ...
            - Horizontal:
              j = 0 -> 1, j = 1 -> 2, j = 3 -> c, ...
        
        Elif top left = h1:
            - Vertical:
              i = 0 -> 1, i = 1 -> 2, ...
            - Horizontal:
              j = 0 -> h, j = 1 -> g, ...
        
        Elif top left = h8
            - Vertical:
              i = 1 -> h, i = 2 -> g, ...
            - Horizontal:
              j = 0 -> 8, j = 1 -> 7, ...
        
        # Maybe just hardcord these maps once?


        """
        warped_image = cv2.imread(warped_image_path)
        # We loop over the pixel coordinates of the bottom left corner of different squares.
        # We start at the bottom left corner of the top-left square.
        # Then via the outer for-loop we increment the vertical coordinate.
            # Note that starting at the top-left corner of the chessboard, to reach the
            # top left coordinate of each square, we need to subtract the height of a square
            # 0, 1, ..., 7 times. Index i represents the nuber of times we subtract.
        # Via the inner for-loop we increment the horizontal coordinate.
            # Note that starting at the top-left corner of the chessboard,
            # to reach the x-coordinate that corresponds to the
            # left side of the next-left square, we need to add the pixel width of a square
            # 0, 1, ..., 7 times. Index j represents the number of times we add.
        # Depending on which corner of the chess board is at the top-left corner of the image
        # from the robot's perspective, the map from indices i and j (which uniquely identify)
        # a square on the image since they identify the bottom-left pixel coordinate of that square)
        # to the square label (e.g. "e2") differs. 
        # Here we just manually specify it for the four options.
        files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        ranks = [str(i) for i in range(1, 9)]
        label_map = {
            "a8": [
                {i: ranks[-(i + 1)] for i in range(8)}, # map from i index to rank of chess square
                {j: files[j] for j in range(8)} # rank from j index to file of chess square
            ],
            "a1": [
                {i: files[i] for i in range(8)}, # note that in this case i determines the file 
                {j: ranks[j] for j in range(8)}
            ],
            "h1": [
                {i: ranks[i] for i in range(8)},
                {j: files[-(j + 1)] for j in range(8)}
            ],
            "h8": [
                {i: files[-(i + 1)] for i in range(8)},
                {j: ranks[-(j + 1)] for j in range(8)}
            ],
        }
                
        # Boolean for cases where i determines file rather than rank
        is_reverse = self.corner_map["tl"] in ["a1", "h8"]
        label_map = label_map[self.corner_map["tl"]]

        # Square size
        square_size = self.board_size // 8 # this should be multiple of 8 anyways

        # Top left coordinates of top-left square
        tl_y = self.padding["up"]
        tl_x = self.padding["left"]

        squares_folder = warped_image_path.parent / "squares"
        squares_folder.mkdir(exist_ok=True) # exist_ok=True for debugging purposes

        for i in range(8):
            for j in range(8):
                top = tl_y + i * square_size - self.padding["up"]
                bottom = tl_y + (i + 1) * square_size + self.padding["down"]
                left = tl_x + j * square_size - self.padding["left"]
                right = tl_x + (j + 1) * square_size + self.padding["right"]

                square_label = (
                    label_map[1][j] + label_map[0][i] if not is_reverse 
                    else label_map[0][i] + label_map[1][j]
                )

                # Create cutout
                # Cropping works using numpy slices.
                # Axis 0: y, axis 1: x, axis 2: colours
                square_cutout = warped_image[top:bottom, left:right]
                square_cutout_annotated = square_cutout.copy()

                # Corners of the actual board square in the full warped image
                square_left = tl_x + j * square_size
                square_right = tl_x + (j + 1) * square_size
                square_top = tl_y + i * square_size
                square_bottom = tl_y + (i + 1) * square_size

                # Convert global warped-image coordinates to local crop coordinates
                # i.e. coordinates of the warped image
                corners_global = [
                    (square_left, square_top),
                    (square_left, square_bottom),
                    (square_right, square_bottom),
                    (square_right, square_top),
                ]

                for x_global, y_global in corners_global:
                    x_local = x_global - left
                    y_local = y_global - top
                    cv2.circle(
                        square_cutout_annotated,
                        (int(x_local), int(y_local)),
                        6,
                        (0, 0, 255),
                        -1,
                    )
                
                # Square path
                # Folder structure will be:
                    # board setup (with raw.png and metadata)
                    # many snapshots.
                        # snapshot
                        # warped_image
                        # cutouts
                square_folder = squares_folder / square_label
                square_folder.mkdir(exist_ok=True) # exist_ok=True for debugging purposes                

                # Save cutout
                cv2.imwrite(str(square_folder / f"{square_label}.png") , square_cutout)
                cv2.imwrite(str(square_folder / f"{square_label}_annotated.png"), square_cutout_annotated)

if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 4
    # Pass the arguments that Processor class requires when initialised:
        # metadata_path
        # config_path
    # And also pass the argument that warp method requires:
        # image path
    processor = Processor(sys.argv[1], sys.argv[2])
    warped_path = processor.warp(Path(sys.argv[3]))
    processor.cutout(warped_path)
