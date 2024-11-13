"""
Script for capture as usb camera and estimate disparity.
"""

import argparse

import cv2
import numpy as np
import torch

import stereoigev


def resize_by_rate(img, rate=1.0):
    if rate == 1.0:
        return img
    h, w = img.shape[:2]
    new_size = (int(w * rate), int(h * rate))
    return cv2.resize(img, new_size)


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


if __name__ == "__main__":
    import disparity_view
    from disparity_view.o3d_project import gen_tvec, as_extrinsics

    parser = argparse.ArgumentParser(description="disparity tool for ZED2i camera as usb camera")
    parser.add_argument("--calc_disparity", action="store_true", help="calc disparity")
    parser.add_argument("--normal", action="store_true", help="normal map")
    parser.add_argument("--reproject", action="store_true", help="reproject to 2D")
    parser.add_argument("json", help="json file for camera parameter")
    parser.add_argument("--axis", type=int, default=0, help="axis to shift(0; to right, 1: to upper, 2: to far)")
    parser.add_argument("video_num", type=int, help="number in /dev/video")
    args = parser.parse_args()

    view_size_rate = 0.25

    calc_disparity = args.calc_disparity
    normal = args.normal
    reproject = args.reproject
    axis = args.axis
    video_num = int(args.video_num)

    if calc_disparity:
        igev_args = default_args()
        disparity_calculator = stereoigev.DisparityCalculator(args=igev_args)

    if normal:
        converter = disparity_view.DepthToNormalMap()

    cap = cv2.VideoCapture(video_num)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    camera_param = disparity_view.CameraParameter.load_json(args.json)
    stereo_camera = disparity_view.StereoCamera.create_from_camera_param(camera_param)
    expected_height, expected_width = camera_param.height, camera_param.width
    assert stereo_camera.shape[0] == camera_param.height
    assert stereo_camera.shape[1] == camera_param.width
    print(f"{expected_height=}, {expected_width=}")
    scaled_baseline = stereo_camera.scaled_baseline()  # [mm]

    with torch.no_grad():
        while True:
            _, frame = cap.read()
            frame = cv2.resize(frame, (2 * expected_width, expected_height))
            assert frame.shape[0] == expected_height
            assert frame.shape[1] == 2 * expected_width
            H, W = frame.shape[:2]
            half_W = W // 2
            left = frame[:, :half_W, :]
            right = frame[:, half_W:, :]

            shape = left.shape[:2]
            cv2.imshow(f"left and right {shape}", resize_by_rate(frame, view_size_rate))

            if calc_disparity:
                disparity = disparity_calculator.predict(left.copy(), right.copy())
                assert disparity.shape[:2] == left.shape[:2]
                disp = np.round(disparity * 256).astype(np.uint16)
                colored = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=0.01), cv2.COLORMAP_JET)
                cv2.imshow("IGEV", resize_by_rate(colored, view_size_rate))
                if normal:
                    normal_bgr = converter.convert(disp)
                    cv2.imshow("normal", resize_by_rate(normal_bgr, view_size_rate))

                if reproject:
                    stereo_camera.pcd = stereo_camera.generate_point_cloud(disparity, left.copy())
                    baseline = 120.0
                    if axis == 0:
                        tvec = np.array((-baseline, 0.0, 0.0))
                    elif axis == 1:
                        tvec = np.array((0.0, baseline, 0.0))
                    elif axis == 2:
                        tvec = np.array((0.0, 0.0, -baseline))

                    tvec = gen_tvec(scaled_shift=scaled_baseline, axis=axis)
                    extrinsics = as_extrinsics(tvec)
                    projected = stereo_camera.project_to_rgbd_image(extrinsics=extrinsics)
                    reprojected_image = np.asarray(projected.color.to_legacy())

                    # reprojected_image = disparity_view.reproject_from_left_and_disparity(
                    #     left, disparity, camera_matrix, baseline=baseline, tvec=tvec
                    # )
                    cv2.imshow("reprojected", resize_by_rate(reprojected_image, view_size_rate))
            key = cv2.waitKey(100)
            if key == ord("q"):
                exit()
