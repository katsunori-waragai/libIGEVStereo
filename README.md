# libIGEVStereo
module to use IGEVStereo

## what is IGEV Stereo
![](doc/demo-imgs.png)
![](doc/IGEV-Stereo.png)
#### original code 
https://github.com/gangweiX/IGEV

arXiv
[Iterative Geometry Encoding Volume for Stereo Matching](https://arxiv.org/abs/2303.06615)

```

@inproceedings{xu2023iterative,
  title={Iterative Geometry Encoding Volume for Stereo Matching},
  author={Xu, Gangwei and Wang, Xianqi and Ding, Xiaohuan and Yang, Xin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21919--21928},
  year={2023}
}

@article{xu2024igev++,
  title={IGEV++: Iterative Multi-range Geometry Encoding Volumes for Stereo Matching},
  author={Xu, Gangwei and Wang, Xianqi and Zhang, Zhaoxing and Cheng, Junda and Liao, Chunyuan and Yang, Xin},
  journal={arXiv preprint arXiv:2409.00638},
  year={2024}
}

```

###### IGEV++
newer version for IGEV
https://github.com/gangweiX/IGEV-plusplus

## checked environment
- NVIDIA Jetson AGX orin
- Ubuntu 20
- python3.8
- docker
#### Note:
ZED2i camera with ZED SDK is optional.
You don't need them to use stereoigev module.

## install (docker case)
docker_build.sh
docker_run.sh

## download model file 
Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1SsMHRyN7808jDViMN1sKz1Nx-71JxUuz?usp=share_link)

or
```commandline
make download
```

## sample execution
```commandline
#!/bin/sh
python3 igev_for_presaved.py --restore_ckpt ./stereoigev/models/sceneflow.pth -l "test/test-imgs/left/left*.png" -r "test/test-imgs/right/right*.png"
```
 
## how to use the module
- All you have to know is
  - stereoigev.DisparityCalculator
  - stereoigev.as_torch_img
- You can see example in igev_for_presaved.py and usb_cam.py.
- note:
  - `with torch.no_grad():` is important to execute this torch based library.
### Optional: ZED2i 
```commandline
$ python3 usb_cam.py -h
usage: usb_cam.py [-h] [--calc_disparity] video_num

disparity tool for ZED2i camera as usb camera

positional arguments:
  video_num         number in /dev/video

optional arguments:
  -h, --help        show this help message and exit
  --calc_disparity  calc disparity

$ python3 usb_cam.py --calc_disparity 0
```
## npy file viewer and helper script for zed camera(StereoLabs)
- https://github.com/katsunori-waragai/disparity-view
- pip install disparity-viewer
- view_npy enable you to npy files as pseudo-colored images.
- zed_capture will make it easy access to zed camera.

## PC NVIDIA GPU case
You can port into PC NVIDIA GPU case.
All you have to do is change original docker image written in Dockerfile.

```commandline
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
```

## Evaluation & Training
- see Evaluation or Training in the original github
https://github.com/gangweiX/IGEV
- This repository does not provide such tools.

## OpenCV version
- OpenCV is used only to IO including VideoCapture(), imshow(), and color mapping.
- You can change opencv-python version.
- Be careful some version has circular import error.
