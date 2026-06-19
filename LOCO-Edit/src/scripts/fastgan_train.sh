#!/bin/bash
# fastgan_train.sh  — FastGAN (lightweight-gan)

export CUDA_VISIBLE_DEVICES=9

DATA_DIR="data/microscopy_lora_new"     # run from src/ ; 512px RGB tiles + metadata.jsonl (ignored)
RESULTS="checkpoints_new/fastgan_microscopy"

lightweight_gan \
  --data            "$DATA_DIR" \
  --results_dir     "$RESULTS/results" \
  --models_dir      "$RESULTS/models" \
  --name            microscopy \
  --image-size      256 \
  --batch-size      16 \
  --gradient-accumulate-every 1 \
  --num-train-steps 10000 \
  --save-every      2000 \
  --evaluate-every  2000 \
  --aug-prob        0.5 \
  --aug-types       "[translation,cutout]" \
  --disc-output-size 1 \
  --attn-res-layers "[32,64]" \
  --calculate-fid-every 2000 \
  --amp