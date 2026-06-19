#!/bin/bash
# main_microscopy_lora.sh
export CUDA_VISIBLE_DEVICES=9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
    --sh_file_name          channel_main.sh                         \
    --device                cuda:0                                          \
    --model_name            Manojb/stable-diffusion-2-1-base                \
    --dataset_name          Examples_2                                        \
    --sample_idx            0                                               \
    --for_prompt            "an image of cells in fluroscent microscopy"    \
    --edit_prompt           ""                                              \
    --inv_prompt            "an image of cells in fluroscent microscopy"    \
    --neg_prompt            ""                                              \
    --lora_path             checkpoints_new/sd21_chA              \
    --mask_path             ""                             \
    --x_space_guidance_scale        0.35                                    \
    --x_space_guidance_num_step     16                                      \
    --edit_t                        0.5                                     \
    --run_edit_null_space_projection_zt     True                            \
    --non_semantic                  True                                    \
    --note                          "channel_A"                  \
    --guidance_scale                1.5                                     \
    --guidance_scale_edit           1.0                                     \
    --seed                          42                                      \
    --null_space_projection         False                                    \
    --pca_rank_null                 5                                       \
    --pca_rank                      5                                       \
    --sampling_mode                 False                                   \
    --tilda_v_score_type            "null+(for-null)"                       \
    --dtype                         fp32                                    \
    --cache_folder                  /scratch/pbk5339/caches/hf/hub          \
    --vis_num                       2                                       \
    --use_sega                      False                                   \
    --inv_steps                     50                                      \
    --for_steps                     50