import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from libigev_stereo.igev_stereo import IGEVStereo
from libigev_stereo.utils.utils import InputPadder

DEVICE = "cuda"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "libigev_stereo/pretrained_models/sceneflow/sceneflow.pth"

assert DEFAULT_MODEL.is_file()


def as_torch_img(numpy_img: np.ndarray, is_BGR_order=True):
    if numpy_img.shape[2] == 4:
        numpy_img = numpy_img[:, :, :3]
    if is_BGR_order:
        numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(numpy_img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    return as_torch_img(img, is_BGR_order=False)


@dataclass
class DisparityCalculator:
    """
    A class to calculate disparity.

    The original code is in
        https://github.com/gangweiX/IGEV

    calc_by_torch_image(self, torch_image1, torch_image2) -> np.ndarray
    """

    args: argparse.Namespace = field(default=None)
    model: torch.nn.DataParallel = field(default=None)

    def __post_init__(self):
        self.model = torch.nn.DataParallel(IGEVStereo(self.args), device_ids=[0])
        self.model.load_state_dict(torch.load(self.args.restore_ckpt))

        self.model = self.model.module
        self.model.to(DEVICE)

        self.model.eval()

    def calc_by_name(self, leftname, rightname) -> np.ndarray:
        torch_image1 = load_image(leftname)
        torch_image2 = load_image(rightname)
        return self.calc_by_torch_image(torch_image1, torch_image2)

    def calc_by_torch_image(self, torch_image1, torch_image2) -> np.ndarray:
        padder = InputPadder(torch_image1.shape, divis_by=32)
        torch_image1, torch_image2 = padder.pad(torch_image1, torch_image2)
        disp = self.model(torch_image1, torch_image2, iters=self.args.valid_iters, test_mode=True)
        disp = disp.cpu().numpy()
        disp = padder.unpad(disp)
        disparity = disp.squeeze()
        return disparity

    def calc_by_bgr(self, bgr1: np.ndarray, bgr2: np.ndarray) -> np.ndarray:
        torch_image1 = as_torch_img(bgr1, is_BGR_order=True)
        torch_image2 = as_torch_img(bgr2, is_BGR_order=True)
        return self.calc_by_torch_image(torch_image1, torch_image2)
