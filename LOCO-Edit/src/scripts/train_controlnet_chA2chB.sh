#!/bin/bash
export CUDA_VISIBLE_DEVICES=9

accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path  checkpoints_new/sd21_chB_fused          \
  --train_data_dir                 data/controlnet_pairs                   \
  --image_column                   image                                   \
  --conditioning_image_column      conditioning_image                      \
  --caption_column                 text                                    \
  --resolution                     512                                     \
  --train_batch_size               2                                       \
  --gradient_accumulation_steps    4                                       \
  --gradient_checkpointing                                                 \
  --mixed_precision                fp16                                    \
  --learning_rate                  1e-5                                    \
  --max_train_steps                4000                                    \
  --checkpointing_steps            2000                                    \
  --validation_steps               1000                                    \
  --validation_image               "data/microscopy_lora_chA/v00_f000_t0_chA.png" \
  --validation_prompt              "an image of cells in fluorescent microscopy" \
  --output_dir                     checkpoints_new/controlnet_chA2chB      \
  --num_validation_images  1 \
  --enable_xformers_memory_efficient_attention                         \
  --cache_dir  /scratch/pbk5339/caches/hf/hub  \
  --seed                           42