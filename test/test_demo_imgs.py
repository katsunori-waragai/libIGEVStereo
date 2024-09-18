"""
sample script for IGEV Stereo
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from libigev_stereo.lib_disparity import DisparityCalculator


def demo(args):
    disparity_calculator = DisparityCalculator(args=args)
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for imfile1, imfile2 in tqdm(list(zip(left_images, right_images))):
            disparity = disparity_calculator.calc_by_name(imfile1, imfile2)
            file_stem = imfile1.split("/")[-2]
            filename = output_directory / f"{file_stem}.png"

            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disparity)
            disp = np.round(disparity * 256).astype(np.uint16)
            cv2.imwrite(
                str(filename),
                cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=0.01), cv2.COLORMAP_JET),
                [int(cv2.IMWRITE_PNG_COMPRESSION), 0],
            )
            print(f"saved {filename}")


def test_all():
    from argparse import Namespace

    args = Namespace(
        corr_implementation="reg",
        corr_levels=2,
        corr_radius=4,
        hidden_dims=[128, 128, 128],
        left_imgs="test-imgs/PlaytableP/im0.png",
        max_disp=192,
        mixed_precision=False,
        n_downsample=2,
        n_gru_layers=3,
        output_directory="./test-output/",
        restore_ckpt="../libigev_stereo/models/sceneflow.pth",
        right_imgs="test-imgs/PlaytableP/im1.png",
        save_numpy=True,
        shared_backbone=False,
        slow_fast_gru=False,
        valid_iters=32,
    )

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)
    print(f"{args=}")
    demo(args)
    assert Path("./test-output/").is_dir()
    assert list(Path("./test-output/").glob("*.png"))


if __name__ == "__main__":
    test_all()
