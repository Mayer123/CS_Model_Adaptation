#!/usr/bin/env bash

#For BART
CUDA_VISIBLE_DEVICES=0 python run_autoprompt.py \
    --data_dir ../../Data/protoQA/ \
    --outdir bart_autoprompt \
    --bsz 64 \
    --eval-size 64 \
    --accumulation-steps 2 \
    --model-name bart \
    --seed 15213 \
    --num-cand 100 \
    --num_triggers 10 \
    --do_train \
    --num_epochs 1 \
    --train_name _non-overlap \
    --max_steps 351
CUDA_VISIBLE_DEVICES=0 python run_autoprompt_generation.py \
    --model_type bart \
    --model_name_or_path bart_autoprompt \
    --length=10 \
    --num_samples=300 \
    --temperature=0.69 \
    --input_file='../../Data/protoQA/dev.crowdsourced.jsonl' \


# For GPT2
# CUDA_VISIBLE_DEVICES=0 python run_autoprompt.py \
#     --data_dir ../../Data/protoQA/ \
#     --outdir gpt2_autoprompt \
#     --bsz 32 \
#     --eval-size 64 \
#     --accumulation-steps 4 \
#     --model-name gpt2 \
#     --seed 15213 \
#     --num-cand 100 \
#     --num_triggers 10 \
#     --do_train \
#     --num_epochs 1 \
#     --initial-trigger 'Based on simple commonsense fact, we know that' \
#     --train_name _similarity \
#     --max_steps 351 
# CUDA_VISIBLE_DEVICES=0 python run_autoprompt_generation.py \
#     --model_type gpt2 \
#     --model_name_or_path gpt2_autoprompt \
#     --length=10 \
#     --num_samples=300 \
#     --temperature=0.69 \
#     --input_file='../../Data/protoQA/dev.crowdsourced.jsonl' \


# For CommonGen
# CUDA_VISIBLE_DEVICES=0 python run_commongen_autoprompt.py \
#     --data_dir ../../Data/CommonGen/ \
#     --outdir commongen_autoprompt \
#     --bsz 32 --eval-size 32 \
#     --accumulation-steps 4 \
#     --model-name bart \
#     --seed 15213 \
#     --num-cand 10 \
#     --num_triggers 10 \
#     --num_epochs 2 \
#     --do_train \
#     --train_name _min-overlap --max_steps 1052