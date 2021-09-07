#!/usr/bin/env python3
# coding=utf-8
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
Modified from hugging face example code.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange, tqdm
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BartForConditionalGeneration, BartTokenizer
import json
import nltk
from nltk.corpus import stopwords
from protoqa_evaluator.data_processing import load_question_answer_clusters_from_jsonl
from protoqa_evaluator.evaluation import general_eval, evaluate
from protoqa_evaluator.scoring import wordnet_score
from functools import partial
from pathlib import Path
import os
import glob
import string
import unicodedata
import sys
sys.path.append('../')
from Finetune.run_generation import exact_match_all_eval_funcs, set_seed, load_data_from_jsonl, sample_sequence, transform_question, get_question, normalize, eval_model, prepare_input
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'bart': (BartForConditionalGeneration, BartTokenizer),
}

def read_json(filename):
    with open(filename, 'r') as f:
        loaded_json = json.load(f)
    return loaded_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " )
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.69,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stop_token', type=str, default=".",
                        help="Token at which text generation is stopped")
    parser.add_argument('--input_file', type=str, default="./dev.jsonl",
                        help="input file containing sentences")
    parser.add_argument('--output', type=str, default="./",
                        help="input file containing sentences")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    input_filename = args.input_file
    true_dev = load_data_from_jsonl(input_filename)
    qidx, questions = get_question(true_dev)

    en_stopwords = set(stopwords.words('english'))
    en_stopwords.add('they\'re')

    set_seed(args)
    args.model_type = args.model_type.lower()
    
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.model_type == 'bart':
        tokenizer = tokenizer_class.from_pretrained('facebook/bart-large', cache_dir = args.cache_dir)
        model = model_class.from_pretrained('facebook/bart-large', cache_dir = args.cache_dir)
    elif args.model_type == 'gpt2':
        tokenizer = tokenizer_class.from_pretrained('gpt2-large', cache_dir = args.cache_dir)
        model = model_class.from_pretrained('gpt2-large', cache_dir = args.cache_dir)
        special_tokens_dict = {'pad_token': '<PAD>', 'mask_token': '<MASK>'}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        assert tokenizer.pad_token == '<PAD>'

    model.to(args.device)
    model.eval()

    question_data = load_question_answer_clusters_from_jsonl(args.input_file)
    all_prompts = read_json(os.path.join(args.model_name_or_path, 'trigger_ids_checkpoints.json'))

    all_results = []
    for i, prompt in enumerate(all_prompts):
        prompt = prompt[0]
        print (prompt)
        print ('Evaluating checkpoint ', ' '.join(tokenizer.convert_ids_to_tokens(prompt)))
        set_seed(args)
        if args.eval_all_checkpoints:
            checkpoint = os.path.join(args.model_name_or_path, str(i))
        else:
            checkpoint = args.model_name_or_path
        if not os.path.exists(checkpoint):
            os.mkdir(checkpoint)
        with open(os.path.join(checkpoint, 'prompt.json'), 'w') as fout:
            fout.write(json.dumps(prompt)+'\n')
            fout.write(json.dumps(tokenizer.convert_ids_to_tokens(prompt))+'\n')
        curr_results = eval_model(args, tokenizer, model, qidx, questions, checkpoint, en_stopwords, question_data, prompt=prompt)
        curr_results['step'] = i
        all_results.append(curr_results)
    all_results = sorted(all_results, key=lambda x: x['step'])
    if args.eval_all_checkpoints:
        res_location = os.path.join(args.model_name_or_path, 'all_checkpoints_results.jsonl')
    else:
        res_location = os.path.join(args.model_name_or_path, args.output+'results.jsonl')
    with open(res_location, 'w') as fout:
        for line in all_results:
            fout.write(json.dumps(line)+'\n')

if __name__ == '__main__':
    main()

