#!/usr/bin/env bash

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$(realpath -m "$SCRIPT_DIR/../pretrained_models/Wan2.1-T2V-1.3B-Diffusers")"
PROMPT_CACHE="$(realpath -m "$SCRIPT_DIR/../pretrained_models/prompt_embeddings/wan2.1-t2v-1.3b")"
cd "$SCRIPT_DIR"

MODEL_ARGS=(
    --model_path "$MODEL_PATH"
    --model_name "wan-s1"
    --model_type "real-sr"
    --training_type "sft"
)

OUTPUT_ARGS=(
    --output_dir "checkpoint/DOVE-Wan-s1"
    --report_to "wandb"
)

DATA_ARGS=(
    --data_root "../datasets/train"
    --video_column "../datasets/train/HQ-VSR.txt"
    --train_resolution "25x320x640"
)

TRAIN_ARGS=(
    --train_epochs 1000
    --train_steps 10000
    --seed 42
    --batch_size 2
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"
    --learning_rate 2e-5
    --gradient_checkpointing true
    --max_grad_norm 0.1
    --lr_scheduler "constant_with_warmup"
)

SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory true
    --nccl_timeout 1800
    --stastic_frequency 500
)

CHECKPOINT_ARGS=(
    --checkpointing_steps 1000
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
    --degradation_config "configs/degradation.yaml"
)

accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${SR_ARGS[@]}"
