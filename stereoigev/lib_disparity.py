"""
Wrapper library introduced in forked version.

"""

import argparse
import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from stereoigev.igev_stereo import IGEVStereo
from stereoigev.utils import InputPadder

DEVICE = "cuda"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def as_torch_img(numpy_img: np.ndarray, is_BGR_order=True):
    """
    convert numpy_img to torch_image
    """
    if numpy_img.shape[2] == 4:
        numpy_img = numpy_img[:, :, :3]
    if is_BGR_order:
        numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(numpy_img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def load_image(imfile: str):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    return as_torch_img(img, is_BGR_order=False)


@dataclass
class DisparityCalculator:
    """
    A class to calculate disparity.

    The original code is in
        https://github.com/gangweiX/IGEV

    typical usage:
    disparity_calculator = DisparityCalculator(args=args)

    predict_by_torch_image(self, torch_image1, torch_image2) -> np.ndarray
    """

    args: argparse.Namespace = field(default=None)
    model: torch.nn.DataParallel = field(default=None)

    def __post_init__(self):
        self.model = torch.nn.DataParallel(IGEVStereo(self.args), device_ids=[0])
        self.model.load_state_dict(torch.load(self.args.restore_ckpt))

        self.model = self.model.module
        self.model.to(DEVICE)

        self.model.eval()

    def predict_by_name(self, leftname, rightname) -> np.ndarray:
        torch_image1 = load_image(leftname)
        torch_image2 = load_image(rightname)
        return self.predict_by_torch_image(torch_image1, torch_image2)

    def predict_by_torch_image(self, torch_image1, torch_image2) -> np.ndarray:
        padder = InputPadder(torch_image1.shape, divis_by=32)
        torch_image1, torch_image2 = padder.pad(torch_image1, torch_image2)
        disp = self.model(torch_image1, torch_image2, iters=self.args.valid_iters, test_mode=True)
        disp = disp.cpu().numpy()
        disp = padder.unpad(disp)
        disparity = disp.squeeze()
        return disparity

    def predict_by_bgr(self, bgr1: np.ndarray, bgr2: np.ndarray) -> np.ndarray:
        torch_image1 = as_torch_img(bgr1, is_BGR_order=True)
        torch_image2 = as_torch_img(bgr2, is_BGR_order=True)
        return self.predict_by_torch_image(torch_image1, torch_image2)


def predict_for_presaved(args: argparse.Namespace):
    """
    save disparity files using left_imgs, right_imgs

    args: in Namespace format
        see details in command line help(-h).
    args.left_ims:
    args.right_imgs:
    args.output_directory:
    """

    disparity_calculator = DisparityCalculator(args=args)
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for imfile1, imfile2 in tqdm(list(zip(left_images, right_images))):
            disparity = disparity_calculator.predict_by_name(imfile1, imfile2)
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
