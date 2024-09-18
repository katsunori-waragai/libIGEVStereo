# libIGEVStereo
module to use IGEVStereo

## what is IGEV Stereo
![](doc/demo-imgs.png)
![](doc/IGEV-Stereo.png)
# Under Development

## checked environment
- NVIDIA Jetson AGX orin
- Ubuntu 20
- python3.8

## install (docker case)
docker_build.sh
docker_run.sh

## install 
```commandline
python3 -m pip install .
```

## download model file
Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1SsMHRyN7808jDViMN1sKz1Nx-71JxUuz?usp=share_link)


## sample execution
```commandline
#!/bin/sh
python3 demo_imgs.py --restore_ckpt ./stereoigev/models/sceneflow.pth -l test/test-imgs/PlaytableP/im0.png -r test/test-imgs/PlaytableP/im1.png
```
 

## npy file viewer
- pip install disparity-viewer をすること
- view_npy コマンドを自作している。
それを使うことで、npyファイルを管理すれば、それで十分の状況を作る。

## Under Development

## original code 
https://github.com/gangweiX/IGEV
