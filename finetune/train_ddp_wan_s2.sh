#!/usr/bin/env bash

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$(realpath -m "$SCRIPT_DIR/checkpoint/DOVE-Wan-s1/ckpt-10000-sft")"
PROMPT_CACHE="$(realpath -m "$SCRIPT_DIR/../pretrained_models/prompt_embeddings/wan2.1-t2v-1.3b")"
cd "$SCRIPT_DIR"

MODEL_ARGS=(
    --model_path "$MODEL_PATH"
    --model_name "wan-s2"
    --model_type "real-sr-image-video"
    --training_type "sft"
)

OUTPUT_ARGS=(
    --output_dir "checkpoint/DOVE-Wan-s2"
    --report_to "wandb"
)

DATA_ARGS=(
    --data_root "../datasets/train"
    --video_column "../datasets/train/HQ-VSR.txt"
    --image_data_root "../datasets/train"
    --image_column "../datasets/train/DIV2K_train_HR.txt"
    --train_resolution "2x320x640"
    --image_ratio 0.8
)

TRAIN_ARGS=(
    --train_epochs 10
    --train_steps 500
    --seed 42
    --batch_size 2
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"
    --learning_rate 5e-6
    --gradient_checkpointing true
    --max_grad_norm 0.1
    --lr_scheduler "constant_with_warmup"
)

SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory true
    --nccl_timeout 1800
    --stastic_frequency 100
)

CHECKPOINT_ARGS=(
    --checkpointing_steps 100
    --checkpointing_limit 3
)

VALIDATION_ARGS=(
    --do_validation true
    --validation_dir "../datasets/test/UDM10"
    --validation_steps 500
    --validation_videos "LQ-Video.txt"
    --validation_ref_videos "GT-Video.txt"
    --gen_fps 8
    --raw_test true
    --num_inference_steps 1
    --eval_metric_list "psnr,ssim,lpips,dists,clipiqa"
)

SR_ARGS=(
    --is_latent false
    --is_cache true
    --empty_prompt true
    --prompt_cache "$PROMPT_CACHE"
    --flow_sigma 0.4
    --degradation_config "configs/degradation_image_video.yaml"
)

PERCEPTUAL_ARGS=(
    --use_perceptual_loss true
    --dists_weight 1.0
    --frame_diff_weight 1.0
)

accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${SR_ARGS[@]}" \
    "${PERCEPTUAL_ARGS[@]}"
