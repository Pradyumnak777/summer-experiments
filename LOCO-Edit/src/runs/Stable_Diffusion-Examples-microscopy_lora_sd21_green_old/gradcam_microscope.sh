#!/bin/bash
# gradcam_microscopy_lora.sh

python gradcam_svd.py \
    --sh_file_name          gradcam_microscope.sh                      \
    --device                cuda:0                                          \
    --model_name            Manojb/stable-diffusion-2-1-base                \
    --dataset_name          Examples                                        \
    --sample_idx            0                                               \
    --for_prompt            "an image of cells in fluroscent microscopy"    \
    --edit_prompt           ""                                              \
    --inv_prompt            "an image of cells in fluroscent microscopy"    \
    --neg_prompt            ""                                              \
    --lora_path             checkpoints_2/sd21_microscopy_lora              \
    --mask_path             masks/mask_green.pt                                 \
    --x_space_guidance_scale        0.5                                    \
    --x_space_guidance_num_step     16                                      \
    --edit_t                        0.35                                     \
    --run_edit_null_space_projection_zt     True                            \
    --non_semantic                  True                                    \
    --note                          "microscopy_lora_sd21"                  \
    --guidance_scale                1.5                                     \
    --guidance_scale_edit           1.0                                     \
    --seed                          42                                      \
    --null_space_projection         False                                    \
    --pca_rank_null                 10                                       \
    --pca_rank                      10                                       \
    --sampling_mode                 False                                   \
    --tilda_v_score_type            "null+(for-null)"                       \
    --dtype                         fp32                                    \
    --cache_folder                  /scratch/pbk5339/caches/hf/hub          \
    --vis_num                       2                                       \
    --use_sega                      False                                   \
    --inv_steps                     50                                      \
    --for_steps                     50                                      \
    # --mask_index                    0