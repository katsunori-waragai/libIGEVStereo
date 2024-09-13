import sys

DEVICE = "cuda"
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import glob
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image

from matplotlib import pyplot as plt
from tqdm import tqdm

from libigev_stereo.igev_stereo import IGEVStereo
from libigev_stereo.utils.utils import InputPadder

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = REPO_ROOT / "libigev_stereo/pretrained_models/sceneflow/sceneflow.pth"


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
    args: Namespace=
    """
    args: argparse.Namespace = field(default=None)
    model: torch.nn.DataParallel = field(default=None)

    def __post_init__(self):
        self.model = torch.nn.DataParallel(IGEVStereo(self.args), device_ids=[0])
        self.model.load_state_dict(torch.load(args.restore_ckpt))

        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()

    def calc_disparity(self, leftimg, rightimg):
        pass

def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for imfile1, imfile2 in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = disp.cpu().numpy()
            disp = padder.unpad(disp)
            file_stem = imfile1.split("/")[-2]
            filename = output_directory / f"{file_stem}.png"
            disparity = disp.squeeze()

            plt.imsave(output_directory / f"{file_stem}.png", disparity, cmap="jet")
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disparity)
            disp = np.round(disp * 256).astype(np.uint16)
            cv2.imwrite(
                str(filename),
                cv2.applyColorMap(cv2.convertScaleAbs(disp.squeeze(), alpha=0.01), cv2.COLORMAP_JET),
                [int(cv2.IMWRITE_PNG_COMPRESSION), 0],
            )
            print(f"saved {filename}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore_ckpt",
        help="restore checkpoint",
        default=DEFAULT_MODEL,
    )
    parser.add_argument("--save_numpy", default=True, help="save output as numpy arrays")

    parser.add_argument("-l", "--left_imgs", help="path to all first (left) frames", default="./demo-imgs/*/im0.png")
    parser.add_argument("-r", "--right_imgs", help="path to all second (right) frames", default="./demo-imgs/*/im1.png")

    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/Middlebury/trainingH/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/Middlebury/trainingH/*/im1.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/ETH3D/two_view_training/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/ETH3D/two_view_training/*/im1.png")
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
    print(f"{args=}")
    demo(args)
