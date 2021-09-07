#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python run_prefix_tuning.py \
    --model_type=bart \
    --model_name_or_path=facebook/bart-large \
    --data_dir=../../Data/protoQA/ \
    --do_train --do_eval --evaluate_during_training \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=64 \
    --gradient_accumulation_steps=1 \
    --logging_steps 500  \
    --output_dir=bart_prefix \
    --num_train_epochs=1 \
    --warmup_steps 150 \
    --save_steps -1 \
    --learning_rate 5e-5 \
    --adam_epsilon 1e-6 --seed 15213 
    # for non-overlap --train_name _non-overlap --max_steps 5621
    # for similarity --train_name _similarity --max_steps 5621

CUDA_VISIBLE_DEVICES=0 python run_prefix_generation.py \
    --model_type bart \
    --model_name_or_path bart_prefix \
    --length=10 \
    --num_samples=300 \
    --temperature=0.69 \
    --input_file='../../Data/protoQA/dev.crowdsourced.jsonl'  


# For CommonGen 
# CUDA_VISIBLE_DEVICES=0 python run_commongen_prefix.py \
#     --model_type=bart \
#     --model_name_or_path=facebook/bart-large \
#     --data_dir ../../Data/CommonGen/ \
#     --do_train --do_eval --evaluate_during_training \
#     --per_gpu_train_batch_size=16 \
#     --per_gpu_eval_batch_size=32 \
#     --gradient_accumulation_steps=1 \
#     --logging_steps 500 \
#     --output_dir bart_prefix_commongen \
#     --num_train_epochs 2 \
#     --warmup_steps 500 \
#     --save_steps -1 \
#     --learning_rate 5e-5 \
#     --adam_epsilon 1e-6 --seed 15213
#     # For min-overlap --train_name _min-overlap --max_steps 8424
#     # For random --train_name _random17K --max_steps 8424