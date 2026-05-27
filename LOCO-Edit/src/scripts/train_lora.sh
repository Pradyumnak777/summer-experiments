#!/bin/bash
# train_lora.sh

export MODEL_NAME="Manojb/stable-diffusion-2-1-base"
export DATA_DIR="data/microscopy_lora"
export OUT_DIR="checkpoints_2/sd21_microscopy_lora"
export TRAIN_SCRIPT="/scratch/pbk5339/summer/diffusers_repo/examples/text_to_image/train_text_to_image_lora.py"
export HF_HOME="/scratch/pbk5339/caches/hf"
export CUDA_VISIBLE_DEVICES=0

accelerate launch --num_processes=1 --mixed_precision="bf16" \
  "$TRAIN_SCRIPT" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --image_column=image \
  --caption_column=text \
  --resolution=512 \
  --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=1500 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --checkpointing_steps=250 \
  --validation_prompt="an image of cells in fluroscent microscopy" \
  --validation_epochs=2 \
  --num_validation_images=2 \
  --seed=42 \
  --output_dir=$OUT_DIR