from pathlib import Path
import cv2
import numpy as np
from omegaconf import OmegaConf
import json

from chess_assistant.camera_utils import build_undistort_maps, undistort

def letterbox(
    img: np.ndarray,
    target_size: tuple[int, int],
    pad_value_img: int | float = 0,
    pad_value_mask: int | float = 0,
) -> np.ndarray:
    """
    Letterbox a 4-channel image to target size.

    Args:
        img: np.ndarray of shape (H, W, 4).
             Channels 0:3 are image channels.
             Channel 3 is a binary mask with values 0 or 1.
        target_size: (target_h, target_w).
        pad_value_img: padding value for image channels.
        pad_value_mask: padding value for mask channel.

    Returns:
        np.ndarray of shape (target_h, target_w, 4).
    """
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"Expected image shape (H, W, 4), got {img.shape}")

    h, w = img.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / w, target_h / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Split image and mask so we can resize them differently
    image_channels = img[:, :, :3]
    mask_channel = img[:, :, 3]

    resized_image = cv2.resize(
        image_channels,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    resized_mask = cv2.resize(
        mask_channel,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # Create padded output
    out = np.empty((target_h, target_w, 4), dtype=img.dtype)
    out[:, :, :3] = pad_value_img
    out[:, :, 3] = pad_value_mask

    # Center the resized image
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2

    out[top:top + new_h, left:left + new_w, :3] = resized_image
    out[top:top + new_h, left:left + new_w, 3] = resized_mask

    return out


def estimate_vanishing_point(
    base_pts: np.ndarray, ext_pts: np.ndarray
) -> tuple[np.ndarray, float]:
    """Least-squares intersection of the N base->ext rays.

    ``base_pts``, ``ext_pts``: ``(N, 2)`` in the same frame. Returns ``(V, mean_residual)`` —
    the point minimising the summed squared perpendicular distance to every ray, and the mean
    perpendicular distance from ``V`` to each ray (a fit-quality diagnostic; one ray far larger
    than the others usually means a bad click, not a modelling problem).
    """
    base_pts = np.asarray(base_pts, dtype=np.float64)
    ext_pts = np.asarray(ext_pts, dtype=np.float64)
    dirs = ext_pts - base_pts
    dirs_unit = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for p, d in zip(base_pts, dirs_unit):
        d = d.reshape(2, 1)
        M = np.eye(2) - d @ d.T
        A += M
        b += M @ p
    V = np.linalg.solve(A, b)
    residuals = [
        np.linalg.norm((np.eye(2) - d.reshape(2, 1) @ d.reshape(1, 2)) @ (V - p))
        for p, d in zip(base_pts, dirs_unit)
    ]
    return V, float(np.mean(residuals))


def bilinear_interp(m_tl, m_tr, m_br, m_bl, u, v):
    """Bilinear value over a unit patch. ``(u, v)`` in ``[0,1]^2`` (u: left->right, v: top->bottom).

    Corners map exactly: ``(0,0)->m_tl``, ``(1,0)->m_tr``, ``(1,1)->m_br``, ``(0,1)->m_bl``.
    """
    top = m_tl * (1 - u) + m_tr * u
    bottom = m_bl * (1 - u) + m_br * u
    return top * (1 - v) + bottom * v


class QuadrantMagnitudeField:
    """Per-square extension magnitude over board coords ``(u, v) in [0,1]^2``.

    Four corner + one centre magnitudes are measured; the four edge midpoints are derived as
    the mean of their two adjacent corners (matching what a plain bilinear patch predicts along
    an edge). The board is split into NW/NE/SE/SW quadrant patches (each bounded by one real
    corner, the centre, and two derived edge midpoints); a query ``(u, v)`` is dispatched to its
    quadrant, remapped to that quadrant's local ``[0,1]^2`` and bilinearly interpolated.
    """

    def __init__(self, m_tl, m_tr, m_br, m_bl, m_center):
        self.m_tl = m_tl
        self.m_tr = m_tr
        self.m_br = m_br
        self.m_bl = m_bl
        self.m_center = m_center
        # Derived edge midpoints.
        self.m_tm = (m_tl + m_tr) / 2  # top
        self.m_rm = (m_tr + m_br) / 2  # right
        self.m_bm = (m_br + m_bl) / 2  # bottom
        self.m_lm = (m_tl + m_bl) / 2  # left

    def __call__(self, u, v):
        u = min(max(float(u), 0.0), 1.0)
        v = min(max(float(v), 0.0), 1.0)
        if u <= 0.5 and v <= 0.5:  # NW
            vals = (self.m_tl, self.m_tm, self.m_center, self.m_lm)
            lu, lv = u * 2, v * 2
        elif u > 0.5 and v <= 0.5:  # NE
            vals = (self.m_tm, self.m_tr, self.m_rm, self.m_center)
            lu, lv = (u - 0.5) * 2, v * 2
        elif u > 0.5 and v > 0.5:  # SE
            vals = (self.m_center, self.m_rm, self.m_br, self.m_bm)
            lu, lv = (u - 0.5) * 2, (v - 0.5) * 2
        else:  # SW
            vals = (self.m_lm, self.m_center, self.m_bm, self.m_bl)
            lu, lv = u * 2, (v - 0.5) * 2
        return bilinear_interp(vals[0], vals[1], vals[2], vals[3], lu, lv)


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
        square_cutout_size = None
        if config_path:
            config = OmegaConf.load(config_path)
            board_size = config.get("image_processing", OmegaConf.create()).get("board_size")
            square_cutout_size = config.get("image_processing", OmegaConf.create()).get("square_cutout_size")
        board_size = board_size if board_size else 400
        square_cutout_size = square_cutout_size if square_cutout_size else 144
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
        # v2 calibration adds a measured "center" extension point (its base is the board
        # centre, derived from the homography at board coord (0.5, 0.5); only its extended
        # position is clicked). Correctness fix vs the old 4-corner-only padding: the measured
        # centre can exceed every corner, so the canvas must be sized over all 5 measured
        # extension points, not just the 4 corners.
        is_v2 = "extended_center_px" in metadata
        differences_for_padding = pixel_differences
        if is_v2:
            dst_extended_center_initial = cv2.perspectiveTransform(
                np.array([[metadata["extended_center_px"]]], dtype=np.float32),
                matrix_initial,
            ).reshape(2)
            center_base_initial = np.array([last_coordinate / 2.0, last_coordinate / 2.0])
            differences_for_padding = np.vstack(
                [pixel_differences, dst_extended_center_initial - center_base_initial]
            )

        # p_up: 'padding_up'. Max signed extension across the measured points in each
        # direction, floored at 0, with a 5% margin.
        p_up = round(1.05 * max(np.max(-differences_for_padding, axis=0)[1], 0))
        p_down = round(1.05 * max(np.max(differences_for_padding, axis=0)[1], 0))
        p_left = round(1.05 * max(np.max(-differences_for_padding, axis=0)[0], 0))
        p_right = round(1.05 * max(np.max(differences_for_padding, axis=0)[0], 0))

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
        self.square_cutout_size = square_cutout_size
        self.image_size = (size_x, size_y)
        self.matrix = matrix

        # Lens undistortion maps, built once per setup from the intrinsics cached in the
        # calibration metadata (v2 calibrations only). When absent (v1 metadata), warp()
        # behaves exactly as before — no undistortion.
        self.undistort_map1 = None
        self.undistort_map2 = None
        camera_intrinsics = metadata.get("camera_intrinsics")
        if camera_intrinsics:
            width, height = camera_intrinsics["image_size"]
            self.undistort_map1, self.undistort_map2 = build_undistort_maps(
                np.asarray(camera_intrinsics["K"], dtype=np.float64),
                np.asarray(camera_intrinsics["D"], dtype=np.float64),
                (width, height),
            )

        # Perspective geometry (v2 calibration only): vanishing point V and the quadrant
        # magnitude field, both expressed in the padded warped frame that cutout() crops from
        # (so V and the per-square floor corners share one frame, no conversion needed).
        self.is_v2 = is_v2
        self.V = None
        self.vp_residual = None
        self.magnitude_field = None
        self.square_geometry = None
        if is_v2:
            base_center = np.array(
                [padding["left"] + board_size / 2.0, padding["up"] + board_size / 2.0]
            )
            base_pts = np.vstack([dst.astype(np.float64), base_center])  # (5, 2), tl,tr,br,bl,c

            ext_corners = (
                cv2.perspectiveTransform(src_extended_corners.reshape(4, 1, 2), matrix)
                .reshape(4, 2)
                .astype(np.float64)
            )
            ext_center = (
                cv2.perspectiveTransform(
                    np.array([[metadata["extended_center_px"]]], dtype=np.float32), matrix
                )
                .reshape(2)
                .astype(np.float64)
            )
            ext_pts = np.vstack([ext_corners, ext_center])  # (5, 2)

            self.V, self.vp_residual = estimate_vanishing_point(base_pts, ext_pts)

            # Extension magnitudes at the 5 measured points (corners in tl, tr, br, bl order).
            corner_mags = np.linalg.norm(ext_corners - dst.astype(np.float64), axis=1)
            self.magnitude_field = QuadrantMagnitudeField(
                m_tl=float(corner_mags[0]),
                m_tr=float(corner_mags[1]),
                m_br=float(corner_mags[2]),
                m_bl=float(corner_mags[3]),
                m_center=float(np.linalg.norm(ext_center - base_center)),
            )

    def warp(self, image_path: Path) -> Path:
        image = cv2.imread(image_path)
        # Undistort in memory before warping. The raw frame on disk is never modified, and no
        # undistorted copy of a full frame is written out (only the warped result is saved).
        if self.undistort_map1 is not None:
            image = undistort(image, self.undistort_map1, self.undistort_map2)
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

        squares_dir = warped_image_path.parent / "squares"
        squares_dir.mkdir(exist_ok=True) # exist_ok=True for debugging purposes

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

                # Now I'm interested in top, bottom, left, right of the actual SQUARE,
                # not the cutout, in the cutout image
                square_left_cutout = self.padding["left"] 
                    # this is pixel coordinate in x direction of the left edge of the square
                    # in the cutout
                    # where the square is located, depends just on the padding
                square_right_cutout = self.padding["left"] + square_size
                square_top_cutout = self.padding["up"]
                square_bottom_cutout = self.padding["up"] + square_size

                 # Add a fourth masking channel
                mask = np.zeros_like(square_cutout[:,:,0])
                mask[square_top_cutout:square_bottom_cutout, square_left_cutout:square_right_cutout] = 1
                square_cutout_masked = np.concatenate([square_cutout, np.expand_dims(mask, mask.ndim)], axis=2)

                # Corners of the actual board square in the full warped image
                square_left = tl_x + j * square_size
                square_right = tl_x + (j + 1) * square_size
                square_top = tl_y + i * square_size
                square_bottom = tl_y + (i + 1) * square_size

                # Letterbox pad the square cutout so it has size self.square_size x self.square_size
                square_cutout_masked =letterbox(
                    square_cutout_masked,
                    (self.square_cutout_size, self.square_cutout_size)
                )

                square_cutout_annotated = square_cutout.copy()

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
                square_dir = squares_dir / square_label
                square_dir.mkdir(exist_ok=True) # exist_ok=True for debugging purposes                

                # Save metadata for square
                square_metadata = {
                    # Save pixel coordinates of top left of square within metadata
                    # so model can learn e.g. that pieces further away from the camera appear larger
                    "top": top,
                    "left": left
                }

                # Save cutout
                cv2.imwrite(str(square_dir / f"{square_label}.png") , square_cutout[:,:,:3])
                cv2.imwrite(str(square_dir / f"{square_label}_annotated.png"), square_cutout_annotated)
                # Save uncompressed numpy array which also contains the mask
                square_cutout_masked[:, :, :3] = cv2.cvtColor(
                    square_cutout_masked[:, :, :3],
                    cv2.COLOR_BGR2RGB,
                )
                np.save(str(square_dir / f"{square_label}_masked.npy"), square_cutout_masked)

                # Save square metadata
                with open(square_dir / f"{square_label}_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(square_metadata, f, indent=2)

        return squares_dir

if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 4
    # Example:
    """
    uv run python -m chess_assistant.image_processing data/raw_images/calibration_metadata.json config.yaml data/raw_images/raw.png
    """
    # Pass the arguments that Processor class requires when initialised:
        # metadata_path
        # config_path
    # And also pass the argument that warp method requires:
        # image path
    processor = Processor(sys.argv[1], sys.argv[2])
    warped_path = processor.warp(Path(sys.argv[3]))
    processor.cutout(warped_path)
