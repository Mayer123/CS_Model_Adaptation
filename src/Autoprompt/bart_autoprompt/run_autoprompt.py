import time
import argparse
import json
import logging
from pathlib import Path
import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import sys
sys.path.append('../')
from utils import TriggerDataset


logger = logging.getLogger(__name__)

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        self._stored_gradient_encoder = None        # For bart, the embedding layer is passed twice 
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        if self._stored_gradient is None:
            self._stored_gradient = grad_out[0]
        else:
            assert self._stored_gradient_encoder is None
            self._stored_gradient_encoder = grad_out[0]

    def get(self):
        return self._stored_gradient, self._stored_gradient_encoder
    
    def clean(self):
        self._stored_gradient = None
        self._stored_gradient_encoder = None


class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """
    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        if isinstance(self._model, BartForConditionalGeneration):
            logits, *_ = self._model(**model_inputs, return_dict=False)
        elif isinstance(self._model, GPT2LMHeadModel):
            logits, *_ = self._model(model_inputs['decoder_input_ids'], return_dict=False)
        else:
            raise ValueError('model not recognized')
        return logits


def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    out = model_inputs.copy()
    input_ids = model_inputs['input_ids']
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
    try:
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
    except RuntimeError:
        filled = input_ids
    out['input_ids'] = filled
    if 'decoder_input_ids' in out:
        decoder_trigger_mask = out.pop('decoder_trigger_mask')
        decoder_input_ids = out['decoder_input_ids']
        decoder_filled = decoder_input_ids.masked_scatter(decoder_trigger_mask, trigger_ids)
        out['decoder_input_ids'] = decoder_filled
    return out

def add_task_specific_tokens(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[T]']
    })
    tokenizer.trigger_token = '[T]'
    tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')

def load_pretrained(args):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    if args.model_name == 'gpt2':
        config = GPT2Config.from_pretrained('gpt2-large', cache_dir=args.cache_dir)
        model = GPT2LMHeadModel.from_pretrained('gpt2-large', cache_dir=args.cache_dir)
        model.eval()
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', cache_dir=args.cache_dir)
        special_tokens_dict = {'pad_token': '<PAD>', 'mask_token': '<MASK>'}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
    elif args.model_name == 'bart':
        config = BartConfig.from_pretrained('facebook/bart-large', cache_dir=args.cache_dir)
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', cache_dir=args.cache_dir)
        model.eval()
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', cache_dir=args.cache_dir)
    add_task_specific_tokens(tokenizer)
    return config, model, tokenizer

def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids

def my_loss_fct(predict_logits, label_ids):
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    lm_loss = loss_fct(predict_logits.view(-1, predict_logits.size(-1)), label_ids.view(-1)).view(label_ids.size())
    nb_tokens = (~torch.eq(label_ids, -100)).sum(dim=1, keepdims=True)
    lm_loss /= nb_tokens
    return lm_loss

def run_model(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading model, tokenizer, etc.')
    config, model, tokenizer = load_pretrained(args)
    model.to(device)
    embeddings = model.get_input_embeddings()
    embedding_gradient = GradientStorage(embeddings)
    predictor = PredictWrapper(model)

    args.template = ' '.join(['[T]']*args.num_triggers)
    print (args.template, len(tokenizer))

    # Obtain the initial trigger tokens and label mapping
    if args.initial_trigger:
        trigger_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(args.initial_trigger))
        logger.info(f'Initial trigger: {args.initial_trigger}')
        logger.info(f'Trigger ids: {trigger_ids}')
        print (len(trigger_ids))
        print (tokenizer.convert_ids_to_tokens(trigger_ids))
        assert len(trigger_ids) == args.num_triggers
    else:
        trigger_ids = [tokenizer.mask_token_id] * args.num_triggers
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
    best_trigger_ids = trigger_ids.clone()
    
    evaluation_fn = my_loss_fct
    loss_fct = torch.nn.CrossEntropyLoss()
    print ('initial_trigger', args.initial_trigger)
    logger.info('Loading datasets')

    type_path = f'train{args.train_name}'
    train_dataset = TriggerDataset(tokenizer, data_dir=args.data_dir, type_path=type_path, prefix=args.template, model_type=args.model_name)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=train_dataset.collate_fn)
    
    dev_dataset = TriggerDataset(tokenizer, data_dir=args.data_dir, type_path='dev', prefix=args.template, model_type=args.model_name)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=dev_dataset.collate_fn)

    args.iters = int(args.num_epochs * len(train_dataset) / args.bsz / args.accumulation_steps)
    if args.max_steps is not None:
        args.iters = args.max_steps
        logger.info(f"Overwriting number iters to : {args.iters}")
    args.save_steps = int(4000 / args.bsz / args.accumulation_steps)
    logger.info(f"Number iters : {args.iters}, save steps : {args.save_steps}")
    logger.info('Evaluating')
    numerator = 0
    denominator = 0
    for model_inputs in tqdm(dev_loader):
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.to(device)
        qids = model_inputs.pop('qids')
        labels = model_inputs.pop('labels')
        with torch.no_grad():
            predict_logits = predictor(model_inputs, trigger_ids)
        numerator += evaluation_fn(predict_logits, labels).sum().item()
        denominator += labels.size(0)
    dev_metric = numerator / (denominator + 1e-13)
    logger.info(f'Dev metric: {dev_metric}')

    best_dev_metric = float('inf')
    # Measure elapsed time of trigger search
    start = time.time()
    trigger_ids_checkpoints = []
    no_improve = 0
    global_step = 0
    for i in tqdm(range(args.iters)):

        logger.info(f'Iteration: {i}')
        logger.info('Accumulating Gradient')
        model.zero_grad()

        train_iter = iter(train_loader)
        averaged_grad = None

        # Accumulate
        for step in range(args.accumulation_steps):
            global_step += 1
            # Shuttle inputs to GPU
            try:
                model_inputs = next(train_iter)
            except:
                logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )
                break
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    model_inputs[k] = v.to(device)
            qids = model_inputs.pop('qids')
            labels = model_inputs.pop('labels')
            predict_logits = predictor(model_inputs, trigger_ids)
            loss = my_loss_fct(predict_logits, labels).sum(dim=1).mean()
            loss.backward()

            grad, grad_encoder = embedding_gradient.get()
            embedding_gradient.clean()
            bsz, _, emb_dim = grad.size()
            selection_mask = model_inputs['decoder_trigger_mask'].unsqueeze(-1)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(bsz, args.num_triggers, emb_dim)
            if grad_encoder is not None:
                encoder_selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
                encoder_grad = torch.masked_select(grad_encoder, encoder_selection_mask)
                encoder_grad = encoder_grad.view(bsz, args.num_triggers, emb_dim)
                grad += encoder_grad
                grad /= 2

            if averaged_grad is None:
                averaged_grad = grad.sum(dim=0) / args.accumulation_steps 
            else:
                averaged_grad += grad.sum(dim=0) / args.accumulation_steps
        logger.info('Evaluating Candidates')
        train_iter = iter(train_loader)

        token_to_flip = random.randrange(args.num_triggers)   
        candidates = hotflip_attack(averaged_grad[token_to_flip],
                                    embeddings.weight,
                                    increase_loss=False,
                                    num_candidates=args.num_cand)

        current_score = 0
        candidate_scores = torch.zeros(args.num_cand, device=device)
        denom = 0
        tmp_curr = []
        tmp_cand = [[] for _ in range(args.num_cand)]
        for step in range(args.accumulation_steps):

            try:
                model_inputs = next(train_iter)
            except:
                logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )
                break
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    model_inputs[k] = v.to(device)
            qids = model_inputs.pop('qids')
            labels = model_inputs.pop('labels')
            predict_logits = predictor(model_inputs, trigger_ids)
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
                eval_metric = evaluation_fn(predict_logits, labels).sum().item()

            # Update current score
            current_score += eval_metric#.sum()
            denom += labels.size(0)
            tmp_curr.append(eval_metric)
            # NOTE: Instead of iterating over tokens to flip we randomly change just one each
            # time so the gradients don't get stale.
            for i, candidate in enumerate(candidates):

                temp_trigger = trigger_ids.clone()
                temp_trigger[:, token_to_flip] = candidate
                with torch.no_grad():
                    predict_logits = predictor(model_inputs, temp_trigger)
                    eval_metric = evaluation_fn(predict_logits, labels).sum().item()

                candidate_scores[i] += eval_metric#.sum()
                tmp_cand[i].append(eval_metric)
       
        # KM: a margin is accounted for every example, as we may have some precision issue
        if ((candidate_scores+denom*0.0001) < current_score).any():
            logger.info('Better trigger detected.')
            best_candidate_score = candidate_scores.min()
            best_candidate_idx = candidate_scores.argmin()
            prev_trigger = trigger_ids.clone()
            trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
            if (prev_trigger == trigger_ids).all():
                print ('prev', prev_trigger)
                print ('curr', trigger_ids)
                print ('toktn_to_flip', token_to_flip)
                print ('candidates', candidates)
                print ('best_idx', best_candidate_idx)
                print ('candidate_score', candidate_scores)
                print ('current_score', current_score)
                print ('tmp_curr', tmp_curr)
                print ('tmp_cand', tmp_cand)
                exit(0)
            logger.info(f'Train metric: {best_candidate_score / (denom + 1e-13): 0.4f}')
        else:
            logger.info('No improvement detected. Skipping evaluation.')
            no_improve += 1
            continue

        logger.info('Evaluating')
        numerator = 0
        denominator = 0
        for model_inputs in tqdm(dev_loader):
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    model_inputs[k] = v.to(device)
            qids = model_inputs.pop('qids')
            labels = model_inputs.pop('labels')
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
            numerator += evaluation_fn(predict_logits, labels).sum().item()
            denominator += labels.size(0)
        dev_metric = numerator / (denominator + 1e-13)

        logger.info(f'Trigger tokens: {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}')
        logger.info(f'Trigger tokens: {tokenizer.convert_tokens_to_ids(tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0)))}')
        logger.info(f'Dev metric: {dev_metric}')

        if dev_metric < best_dev_metric:
            logger.info('Best performance so far')
            best_trigger_ids = trigger_ids.clone()
            best_dev_metric = dev_metric
            no_improve = 0
        else:
            no_improve += 1
        logger.info(f'no improve: {no_improve}')

    trigger_ids_checkpoints.append(best_trigger_ids.cpu().numpy().tolist())
    best_trigger_tokens = tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
    logger.info(f'Best tokens: {best_trigger_tokens}')
    logger.info(f'Best dev metric: {best_dev_metric}')
    with open(os.path.join(args.outdir, 'trigger_ids_checkpoints.json'), 'w') as fout:
        json.dump(trigger_ids_checkpoints, fout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True, help='Train data path')
    parser.add_argument("--train_name", default='', type=str, help="The training file name.")
    parser.add_argument('--template', type=str, help='Template string')
    parser.add_argument('--outdir', type=str, required=True, help='output dir')
    parser.add_argument('--initial-trigger', type=str, default=None, help='Manual prompt')
    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=256, help='Eval size')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of epochs to run trigger search algorithm')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of steps to update the model, specify this will overwrite num epochs')
    parser.add_argument('--accumulation-steps', type=int, default=10)
    parser.add_argument('--model-name', type=str, default='bart',
                        help='Model name passed to HuggingFace AutoX classes.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-cand', type=int, default=10)
    parser.add_argument('--num_triggers', type=int, default=3)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)
    if os.path.exists(args.outdir) and os.listdir(args.outdir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.outdir))
    if args.do_train:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        logging.getLogger().handlers.clear()
        handler = logging.FileHandler(os.path.join(args.outdir, 'train.log'))
        logger.addHandler(handler)
        os.system("cp run_autoprompt.py %s" % os.path.join(args.outdir, 'run_autoprompt.py'))
        os.system("cp ../utils.py %s" % os.path.join(args.outdir, 'utils.py'))
    logger.info("Training/evaluation parameters %s", args)
    run_model(args)
