"""
sample script for IGEV Stereo
original:
    https://github.com/gangweiX/IGEV
"""

import argparse
from pathlib import Path

from stereoigev.lib_disparity import demo

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
