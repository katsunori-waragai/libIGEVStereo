import cv2
import numpy as np

import stereoigev

def default_args():
    args = Namespace(
        corr_implementation="reg",
        corr_levels=2,
        corr_radius=4,
        hidden_dims=[128, 128, 128],
        # left_imgs="test-imgs/left/left*.png",
        max_disp=192,
        mixed_precision=False,
        n_downsample=2,
        n_gru_layers=3,
        output_directory="./test-output/",
        restore_ckpt="./stereoigev/models/sceneflow.pth",
        # right_imgs="test-imgs/right/right*.png",
        save_numpy=True,
        shared_backbone=False,
        slow_fast_gru=False,
        valid_iters=32,
    )
    return args

if __name__ == "__main__":
    from argparse import Namespace

    args = default_args()
    disparity_calculator = stereoigev.DisparityCalculator(args=args)

    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        H, W = frame.shape[:2]
        half_W = W // 2
        left = frame[:, :half_W, :]
        right = frame[:, half_W:, :]

        cv2.imshow("left", left)
        disparity = disparity_calculator.calc_by_bgr(left.copy(), right.copy())
        disp = np.round(disparity * 256).astype(np.uint16)
        colored = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=0.01), cv2.COLORMAP_JET)
        cv2.imshow("IGEV", colored)
        key = cv2.waitKey(100)
        if key == ord("q"):
            exit()
