#!/usr/bin/env python3
# coding=utf-8
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
Modified from hugging face example code.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange, tqdm
from collections import Counter, defaultdict

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
from typing import *
import os
import glob
import string
import unicodedata
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

max_answers = {
    f"Max_Answers_{k}": partial(general_eval, max_pred_answers=k)
    for k in [1, 3, 5, 10]
}
max_incorrect = {
    f"Max_Incorrect_{k}": partial(general_eval, max_incorrect=k)
    for k in [1, 3, 5]
}
exact_match_all_eval_funcs = {**max_answers, **max_incorrect}
wordnet_all_eval_funcs = {
    f"WordNet_{k}": partial(v, score_func=wordnet_score, score_matrix_transformation=np.round)
    for k, v in exact_match_all_eval_funcs.items()
}
exact_match_all_eval_funcs.update(wordnet_all_eval_funcs)

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'bart': (BartForConditionalGeneration, BartTokenizer),
}

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_data_from_jsonl(data_path: Union[Path, str]) -> Dict:
    """
    Load jsonl input data, converting the answers-cleaned item to use frozenset keys.
    :param data_path: path to jsonl data
    :return: question_id indexed Dict
    """
    question_data = dict()
    with open(data_path) as data:
        for q in data:
            q_json = json.loads(q)
            # The following attempts to handle various formats of the data
            if "normalized-question" not in q_json:
                q_json["normalized-question"] = q_json["question"][
                    "normalized"
                ]
            if "metadata" in q_json:
                question_data[q_json["metadata"]["id"]] = q_json
            else:
                question_data[q_json["questionid"]] = q_json
    return question_data

def sample_sequence(model, length, inputs, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu', pad_token_id=None):
    context = torch.tensor(inputs['context_tokens'], dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        if isinstance(model, GPT2LMHeadModel):
            generated = model.generate(input_ids=context, max_length=length+context.size(1), do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, num_return_sequences=num_samples, num_beams=1, pad_token_id=pad_token_id)
        elif isinstance(model, BartForConditionalGeneration):
            decoder_inputs = context[:, :-2]
            generated = model.generate(input_ids=context, max_length=length+2+decoder_inputs.size(1), do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, num_return_sequences=num_samples, num_beams=1, pad_token_id=pad_token_id, use_cache=True, decoder_input_ids=decoder_inputs)
    return generated


def transform_question(question):
    question = question.lower()
    question = question.replace('.', '')
    question = question.replace(':', '')
    question = question.replace('?', '')
    question = question.replace('someone', 'one person')
    question = question.replace('someplace', 'one place')
    if 'name something' in question:
        question = question.replace('name something', 'one thing')
        question += ' is'
    elif 'tell me something' in question:
        question = question.replace('tell me something', 'one thing')
        question += ' is'
    elif 'name a ' in question:
        question = question.replace('name a ', 'one ')
        question += ' is'
    elif 'name an ' in question:
        question = question.replace('name an ', 'one ')
        question += ' is'
    elif 'name' in question:
        question = question.replace('name', '')
        question += ' is'
    elif question.startswith('tell me a '):
        question = question.replace('tell me a ', 'one ')
        question += ' is'
    elif question.startswith('tell me an '):
        question = question.replace('tell me an ', 'one ')
        question += ' is'
    elif question.startswith('what '):
        question = question.replace('what', 'one')
        question += ' is'
    elif question.startswith('give me a '):
        question = question.replace('give me a ', 'one ')
        question += ' is'
    elif question.startswith('tell me '):
        question = question.replace('tell me ', '')
        question += ' is'
    elif 'which' in question:
        question = question.replace('which', 'one')
        question += ' is'
    elif 'what' in question:
        question = question.replace('what', 'one')
        question += ' is'
    elif 'how can you tell' in question:
        question = question.replace('how can you tell', 'one way to tell')
        question += ' is'
    else:
        question = 'Q: '+question +'? A: '
    return question

def get_question(data_dict):
    qidx = []
    questions = []
    for q in data_dict:
        question = data_dict[q]['normalized-question']
        transformed_question = transform_question(question)
        questions.append(transformed_question)
        qidx.append(q)
    return qidx, questions

def normalize(text):
    text = unicodedata.normalize('NFD', text)\
        .encode('ascii', 'ignore')\
        .decode("utf-8")
    return str(text)

def prepare_input(tokenizer, raw_text, model_type, prompt=None):
    inputs = {}
    if model_type == 'gpt2':
        context_tokens = tokenizer.encode(raw_text, add_prefix_space=True)
    elif model_type == 'bart':
        context_tokens = tokenizer.encode(raw_text+' '+tokenizer.mask_token, add_special_tokens=True, add_prefix_space=True)
    if prompt == None:
        inputs['context_tokens'] = context_tokens 
    else:
        if model_type == 'bart':
            inputs['context_tokens'] = context_tokens[:1] + prompt + context_tokens[1:]
        else:
            inputs['context_tokens'] = prompt + context_tokens
    return inputs

def eval_model(args, tokenizer, model, qidx, questions, location, en_stopwords, question_data, prompt=None):
    logger.info(args)
    pad_token_id = tokenizer.pad_token_id
    prediced_dev = defaultdict(list)
    i=0
    num = len(questions)
    for single_question_idx in trange(len(questions)):
        raw_text = questions[single_question_idx]
        i+=1
        inputs = prepare_input(tokenizer, raw_text, args.model_type, prompt=prompt)
        out = sample_sequence(
            model=model,
            inputs=inputs,
            num_samples=args.num_samples,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            is_xlnet=False,
            is_xlm_mlm=False,
            xlm_mask_token=None,
            xlm_lang=None,
            device=args.device, pad_token_id=pad_token_id
        )
        context_tokens = inputs['context_tokens']
        if args.model_type == 'gpt2':
            out = out[:, len(context_tokens):].tolist()
        elif args.model_type == 'bart':
            out = out[:, len(context_tokens)-2:].tolist()
        for o in out:
            text = tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            text = text[: text.find(args.stop_token)+1 if args.stop_token else None]
            text = text.strip()
            if text.endswith('.'):
                text = text[:-1]
            nostop_text_list = [tok for tok in text.split() if tok not in en_stopwords]
            nostop_text = " ".join(nostop_text_list)
            nostop_text = normalize(nostop_text)
            nostop_text = ''.join([c for c in nostop_text if c not in ['"']])
            if nostop_text == '':
                continue
            if '/' in nostop_text:
                nostop_text = nostop_text.split('/')
            else:
                nostop_text = [nostop_text]
            if qidx[single_question_idx] not in prediced_dev:
                prediced_dev[qidx[single_question_idx]] = nostop_text
            else:
                prediced_dev[qidx[single_question_idx]] += nostop_text
        counted_value = Counter(prediced_dev[qidx[single_question_idx]])    
        if len(counted_value) == 0:
            prediced_dev[qidx[single_question_idx]] = ['random']

    ranked_predicted_dev = defaultdict(list)
    sampled_answers = defaultdict(list)
    for q in prediced_dev:
        counted_value = Counter(prediced_dev[q])
        sampled_answers[q] = counted_value
        ranked_list = [pair[0] for pair in counted_value.most_common(10)]
        ranked_predicted_dev[q] = ranked_list
    
    results = {}
    errors = {}
    for name, eval_func in exact_match_all_eval_funcs.items():
        try:
            scores = evaluate(eval_func, question_data, answers_dict=ranked_predicted_dev)
        except:
            continue
        results[name] = "{:.3f}".format(np.mean([x.score for x in scores.values()]))
        if name == 'WordNet_Max_Incorrect_3':
            results['full_ids'] = [x for x in scores.keys()]
            results['full'] = ["{:.3f}".format(scores[x].score) for x in results['full_ids']]
            for qid, preds in ranked_predicted_dev.items():
                matched_answers = []
                overlap_answers = []
                wrong_answers = []
                missing_answers = []
                for o in preds:
                    if o in scores[qid].answer_assignment:
                        if scores[qid].answer_assignment[o] is not None:
                            if scores[qid].answer_assignment[o] == '#####':
                                overlap_answers.append(o)
                            else:
                                matched_answers.append(o)
                        else:
                            wrong_answers.append(o)
                    else:
                        wrong_answers.append(o)
                for k, v in question_data[qid].answer_clusters.items():
                    if k not in scores[qid].answer_assignment.values():
                        missing_answers.append([list(k), v])
                errors[qid] = {'matched':matched_answers, 'overlap':overlap_answers, 'wrong':wrong_answers, 'missing':missing_answers,
                 'score':"{:.3f}".format(scores[qid].score), 'question':question_data[qid].question}

    with open(os.path.join(location, args.output+'ranked_list.jsonl'), 'w') as f:
        for key in ranked_predicted_dev:
            sample_d = {key:ranked_predicted_dev[key]}
            sample_d.update(errors[key])
            json.dump(sample_d, f)
            f.write('\n')
    with open(os.path.join(location, args.output+'all_answers.json'), 'w') as f:
        json.dump(sampled_answers, f)
    return results

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
    parser.add_argument('--input_file', type=str, default=None,
                        help="input file containing sentences")
    parser.add_argument('--output', type=str, default="./",
                        help="additional name added to output file")
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
    elif args.model_type == 'gpt2':
        tokenizer = tokenizer_class.from_pretrained('gpt2-large', cache_dir = args.cache_dir)
        special_tokens_dict = {'pad_token': '<PAD>'}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        assert tokenizer.pad_token == '<PAD>'
    else:
        raise ValueError('model not recoginized')
    question_data = load_question_answer_clusters_from_jsonl(args.input_file)

    if args.eval_all_checkpoints:
        checkpoints = list(os.path.dirname(c) for c in glob.glob(args.model_name_or_path+ '/checkpoint*/'))
    else:
        checkpoints = [args.model_name_or_path]

    all_results = []
    for checkpoint in checkpoints:
        print ('Evaluating checkpoint ', checkpoint)
        if 'checkpoint' in checkpoint and args.eval_all_checkpoints:
            global_step = int(os.path.basename(checkpoint).split('-')[1])
        else:
            global_step = -1
        model = model_class.from_pretrained(checkpoint, cache_dir = args.cache_dir)
        model.to(args.device)
        model.eval()
        if checkpoint in ['facebook/bart-large', 'gpt2-large']:
            if checkpoint == 'gpt2-large':
                model.resize_token_embeddings(len(tokenizer))
            checkpoint = './zero_shot'
        set_seed(args)
        curr_results = eval_model(args, tokenizer, model, qidx, questions, checkpoint, en_stopwords, question_data)
        curr_results['step'] = global_step
        all_results.append(curr_results)
    all_results = sorted(all_results, key=lambda x: x['step'])
    if args.eval_all_checkpoints:
        res_location = os.path.join(args.model_name_or_path, 'all_checkpoints_results.jsonl')
    else:
        if args.model_name_or_path in ['facebook/bart-large', 'gpt2-large']:
            res_location = os.path.join('./zero_shot', args.output+'results.jsonl')
        else:
            res_location = os.path.join(args.model_name_or_path, args.output+'results.jsonl')
    with open(res_location, 'w') as fout:
        for line in all_results:
            fout.write(json.dumps(line)+'\n')

if __name__ == '__main__':
    main()

