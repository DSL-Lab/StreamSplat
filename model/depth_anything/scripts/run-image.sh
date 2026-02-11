#! /bin/bash 
python run.py \
  --encoder vitl \
  --img-path ./example/gt_image_0_7_0.jpg --outdir ./example \
  --input-size 518 \
  --pred-only
#   [--input-size <size>] [--pred-only] [--grayscale]