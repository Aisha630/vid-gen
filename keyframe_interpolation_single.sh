#!/bin/bash
noise_injection_steps=2
noise_injection_ratio=0.5
EVAL_DIR=examples
CHECKPOINT_DIR=/home/iml2/interpolation/svd_keyframe_interpolation/checkpoints/svd_reverse_motion_with_attnflip/svd_reverse_motion_with_attnflip/unet
MODEL_NAME=stabilityai/stable-video-diffusion-img2vid-xt
OUT_DIR=results

mkdir -p $OUT_DIR

# Set specific example directory
example_dir="$EVAL_DIR/horses_xt"
example_name=$(basename $example_dir)
echo $example_name

out_fn=$OUT_DIR/$example_name'.gif'
python keyframe_interpolation.py \
    --frame1_path="./input/horses/horses_0.jpeg" \
    --frame2_path="./input/horses/horses_1.jpeg" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --noise_injection_steps=$noise_injection_steps \
    --noise_injection_ratio=$noise_injection_ratio \
    --out_path=$out_fn \
    --num_inference_steps=20

