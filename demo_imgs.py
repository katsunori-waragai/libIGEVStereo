"""
sample script for IGEV Stereo
original:
    https://github.com/gangweiX/IGEV
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from stereoigev import DisparityCalculator, as_torch_img
def demo(args: argparse.Namespace):
    """
    save disparity files using left_imgs, right_imgs

    args: in Namespace format
        see details in command line help(-h).
    """

    disparity_calculator = DisparityCalculator(args=args)
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for imfile1, imfile2 in tqdm(list(zip(left_images, right_images))):
            bgr1 = cv2.imread(str(imfile1))
            bgr2 = cv2.imread(str(imfile2))

            torch_image1 = as_torch_img(bgr1, is_BGR_order=True)
            torch_image2 = as_torch_img(bgr2, is_BGR_order=True)
            disparity = disparity_calculator.calc_by_torch_image(torch_image1, torch_image2)
            file_stem = Path(imfile1).stem
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

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parent
    DEFAULT_MODEL = REPO_ROOT / "stereoigev/models/sceneflow.pth"
    print(f"{DEFAULT_MODEL=}")

    assert DEFAULT_MODEL.is_file()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore_ckpt",
        help="restore checkpoint",
        default=DEFAULT_MODEL,
    )
    parser.add_argument("--save_numpy", default=True, help="save output as numpy arrays")

    parser.add_argument("-l", "--left_imgs", help="path to all first (left) frames", default="./demo-imgs/*/im0.png")
    parser.add_argument("-r", "--right_imgs", help="path to all second (right) frames", default="./demo-imgs/*/im1.png")

    parser.add_argument("--output_directory", help="directory to save output", default="./demo-output/")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--valid_iters", type=int, default=32, help="number of flow-field updates during forward pass")

    # Architecture choices
    parser.add_argument(
        "--hidden_dims", nargs="+", type=int, default=[128] * 3, help="hidden state and context dimensions"
    )
    parser.add_argument(
        "--corr_implementation",
        choices=["reg", "alt", "reg_cuda", "alt_cuda"],
        default="reg",
        help="correlation volume implementation",
    )
    parser.add_argument(
        "--shared_backbone", action="store_true", help="use a single backbone for the context and feature encoders"
    )
    parser.add_argument("--corr_levels", type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument("--corr_radius", type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument("--n_downsample", type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument("--slow_fast_gru", action="store_true", help="iterate the low-res GRUs more frequently")
    parser.add_argument("--n_gru_layers", type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument("--max_disp", type=int, default=192, help="max disp of geometry encoding volume")

    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)
    demo(args)
