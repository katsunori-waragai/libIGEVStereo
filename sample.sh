#!/bin/sh
python3 igev_for_presaved.py --restore_ckpt ./stereoigev/models/sceneflow.pth -l "test/test-imgs/left/left*.png" -r "test/test-imgs/right/right*.png"
 
