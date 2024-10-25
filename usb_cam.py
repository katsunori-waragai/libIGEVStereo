"""
Script for capture as usb camera and estimate disparity.
"""

import argparse

import cv2
import numpy as np
import torch

import stereoigev


def default_args():
    args = argparse.Namespace(
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


def resize_image(frame: np.ndarray) -> np.ndarray:
    H, W = frame.shape[:2]
    return cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    import disparity_view
    from disparity_view.util import dummy_camera_matrix

    parser = argparse.ArgumentParser(description="disparity tool for ZED2i camera as usb camera")
    parser.add_argument("--calc_disparity", action="store_true", help="calc disparity")
    parser.add_argument("--normal", action="store_true", help="normal map")
    parser.add_argument("--reproject", action="store_true", help="reproject to 2D")
    parser.add_argument("video_num", help="number in /dev/video")
    real_args = parser.parse_args()

    calc_disparity = real_args.calc_disparity
    normal = real_args.normal
    reproject = real_args.reproject
    video_num = int(real_args.video_num)

    if calc_disparity:
        args = default_args()
        disparity_calculator = stereoigev.DisparityCalculator(args=args)

    if normal:
        converter = disparity_view.DepthToNormalMap()

    cap = cv2.VideoCapture(video_num)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    with torch.no_grad():
        while True:
            _, frame = cap.read()
            frame = resize_image(frame)
            H, W = frame.shape[:2]
            half_W = W // 2
            left = frame[:, :half_W, :]
            right = frame[:, half_W:, :]

            cv2.imshow("left and right", frame)

            if calc_disparity:
                disparity = disparity_calculator.predict(left.copy(), right.copy())
                disp = np.round(disparity * 256).astype(np.uint16)
                colored = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=0.01), cv2.COLORMAP_JET)
                cv2.imshow("IGEV", colored)
                if normal:
                    normal_bgr = converter.convert(disp)
                    cv2.imshow("normal", normal_bgr)

                if reproject:
                    camera_matrix = dummy_camera_matrix(left.shape)
                    baseline = 120.0
                    tvec = np.array((-baseline, 0.0, 0.0))
                    reprojected_image = disparity_view.reproject_from_left_and_disparity(
                        left, disparity, camera_matrix, baseline=baseline, tvec=tvec
                    )
                    cv2.imshow("reprojected", reprojected_image)
            key = cv2.waitKey(100)
            if key == ord("q"):
                exit()
