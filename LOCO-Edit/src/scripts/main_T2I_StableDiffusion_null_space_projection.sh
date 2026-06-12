python main.py \
    --sh_file_name                          main_T2I_StableDiffusion_null_space_projection.sh           \
    --device                                cuda:9                                      \
    --model_name                            Manojb/stable-diffusion-2-1-base                      \
    --mask_model_name                       facebook/sam-vit-large                      \
    --dataset_name                          Examples_2                                      \
    --edit_prompt                           "a black tiger"                  \
    --for_prompt                            "a tiger"                          \
    --neg_prompt                            ""                                          \
    --x_space_guidance_scale                2                                         \
    --x_space_guidance_num_step             16                                          \
    --edit_t                                0.5                                         \
    --run_edit_null_space_projection_zt_semantic     True                                        \
    --note                                  "with_prompt"                               \
    --guidance_scale                        6.0                                         \
    --guidance_scale_edit                   4.0                                         \
    --seed                                  42                                  \
    --null_space_projection                 False                                        \
    --pca_rank_null                         3                                           \
    --pca_rank                              1                                           \
    --sampling_mode                         False                                       \
    --mask_index                            0                                           \
    --tilda_v_score_type                    "null+(for-null)+(edit-null)"                          \
    --dtype                                 fp32                                        \
    --vis_num                               3                                           \
    --use_sega                              False       \
    --cache_folder                          /scratch/pbk5339/caches/hf/hub         

  