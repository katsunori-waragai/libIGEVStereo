"""
test script for IGEV Stereo
"""

from pathlib import Path


from stereoigev.lib_disparity import predict_for_presaved


def test_all():
    from argparse import Namespace

    args = Namespace(
        corr_implementation="reg",
        corr_levels=2,
        corr_radius=4,
        hidden_dims=[128, 128, 128],
        left_imgs="test-imgs/left/left*.png",
        max_disp=192,
        mixed_precision=False,
        n_downsample=2,
        n_gru_layers=3,
        output_directory="./test-output/",
        restore_ckpt="../stereoigev/models/sceneflow.pth",
        right_imgs="test-imgs/right/right*.png",
        save_numpy=True,
        shared_backbone=False,
        slow_fast_gru=False,
        valid_iters=32,
    )

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)
    print(f"{args=}")
    predict_for_presaved(args)
    assert Path("./test-output/").is_dir()
    assert list(Path("./test-output/").glob("*.png"))
