#!/bin/sh
python3 demo_imgs.py --restore_ckpt ./stereoigev/models/sceneflow.pth -l "test/test-imgs/left/left*.png" -r "test/test-imgs/Pright/right*.png"
 
