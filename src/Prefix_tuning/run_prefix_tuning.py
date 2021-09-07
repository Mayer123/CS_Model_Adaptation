from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn 

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  GPT2Config, GPT2Tokenizer,
                                  BartConfig, BartTokenizer,)
import transformers
transformers.logging.set_verbosity_info()
import sys
sys.path.append('../')
from utils import Seq2SeqDataset
from modeling_bart import MyBartForConditionalGeneration
from modeling_gpt2 import MyGPT2LMHeadModel
from modeling_prefix import PrefixModelBart, PrefixModelGPT2
from Finetune.run_fine_tuning import load_and_cache_examples, set_seed, count_parameters
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, MyGPT2LMHeadModel, GPT2Tokenizer),
    'bart': (BartConfig, MyBartForConditionalGeneration, BartTokenizer),
}

def model_step(batch: dict, model, core_model, tokenizer):
    source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
    if isinstance(core_model, MyBartForConditionalGeneration):
        decoder_input_ids = target_ids[:, :-1].contiguous()
        lm_labels = target_ids[:, 1:].clone()
        for i in range(len(lm_labels)):
            for j in range(len(lm_labels[i])):
                if lm_labels[i][j] == source_ids[i][j+1]:
                    lm_labels[i][j] = -100
                else:
                    break
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100
    elif isinstance(core_model, MyGPT2LMHeadModel):
        decoder_input_ids = target_ids[:, :-1].contiguous()
        lm_labels = target_ids.clone()[:, 1:].clone()
        for i in range(len(lm_labels)):
            for j in range(1, len(source_ids[i])):
                if lm_labels[i][j-1] == source_ids[i][j]:
                    lm_labels[i][j-1] = -100
                else:
                    break
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100

    assert decoder_input_ids is not None
    nb_tokens = (lm_labels != -100).sum(dim=1)
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    if isinstance(core_model, MyBartForConditionalGeneration):
        outputs = model(
            core_model,
            source_ids,
            attention_mask=source_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
    elif isinstance(core_model, MyGPT2LMHeadModel):
        outputs = model(core_model, decoder_input_ids)
    token_wise_loss = ce_loss(outputs['logits'].view(-1, outputs['logits'].size(-1)), lm_labels.view(-1))
    token_wise_loss = token_wise_loss.view(lm_labels.shape)
    token_wise_loss = token_wise_loss.sum(dim=1)
    token_wise_loss = token_wise_loss/nb_tokens
    loss = token_wise_loss.mean(dim=0)
    return (loss,)

def train(args, train_dataset, model, core_model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'runs'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    print (args.save_steps)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    best_loss = 99999
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(args.device)
            model.train()

            outputs = model_step(batch, model, core_model, tokenizer)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, core_model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        if results['eval_loss'] < best_loss:
                            logger.info("Saving model checkpoint to %s", args.output_dir)
                            torch.save(model.state_dict(), os.path.join(args.output_dir, 'prefix_model.bin'))
                            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                            best_loss = results['eval_loss']
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                    torch.save(model.state_dict(), os.path.join(output_dir, 'prefix_model.bin'))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.local_rank in [-1, 0] and args.logging_steps > 0:
            if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                results = evaluate(args, model, core_model, tokenizer)
                for key, value in results.items():
                    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                if results['eval_loss'] < best_loss:
                    logger.info("Saving model checkpoint to %s", args.output_dir)
                    torch.save(model.state_dict(), os.path.join(args.output_dir, 'prefix_model.bin'))
                    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                    best_loss = results['eval_loss']
            tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
            tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
            logging_loss = tr_loss
        if args.local_rank in [-1, 0] and args.save_steps > 0:
            checkpoint_prefix = 'checkpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, 'prefix_model.bin'))
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, core_model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate_fn)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(args.device)
        with torch.no_grad():
            outputs = model_step(batch, model, core_model, tokenizer)
            lm_loss = outputs[0]
            eval_loss += lm_loss.sum().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    result = {
        "eval_loss": eval_loss,
        "perplexity": perplexity
    }
    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_name", default='', type=str, 
                        help="The training file name.")

    ## Other parameters
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument('--num_prefix', type=int, default=10,
                        help="number of prefix tokens for the model.")
    parser.add_argument('--mid_dim', type=int, default=512,
                        help="the middle layer dimension")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_src_len", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--max_tgt_len", default=512, type=int,
                        help="Optional target sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    if args.do_train:
        handler = logging.FileHandler(os.path.join(args.output_dir, 'train.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        os.system("cp run_prefix_tuning.py %s" % os.path.join(args.output_dir, 'run_prefix_tuning.py'))
        os.system("cp run_prefix_generation.py %s" % os.path.join(args.output_dir, 'run_prefix_generation.py'))
        os.system("cp modeling_prefix.py %s" % os.path.join(args.output_dir, 'modeling_prefix.py'))
        os.system("cp modeling_bart.py %s" % os.path.join(args.output_dir, 'modeling_bart.py'))
        os.system("cp modeling_gpt2.py %s" % os.path.join(args.output_dir, 'modeling_gpt2.py'))
        os.system("cp ../utils.py %s" % os.path.join(args.output_dir, 'utils.py'))
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    
    if args.model_type == 'bart':
        prefix_model = PrefixModelBart(args.num_prefix, config, args.mid_dim, args.device)
    elif args.model_type == 'gpt2':
        prefix_model = PrefixModelGPT2(args.num_prefix, config, args.mid_dim, args.device)
    else:
        raise ValueError('model not recognized')
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(args.device)
    prefix_model.to(args.device)
    print (count_parameters(prefix_model), count_parameters(model))

    #add padding token to gpt2
    if args.model_type == 'gpt2':
        special_tokens_dict = {'pad_token': '<PAD>'}

        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print('We have added', num_added_toks, 'tokens')
        model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

        assert tokenizer.pad_token == '<PAD>'
        #add padding token to gpt2

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache   
        if args.local_rank == 0:
            torch.distributed.barrier()
        
        global_step, tr_loss = train(args, train_dataset, prefix_model, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            if not args.do_train:
                prefix_model.load_state_dict(torch.load(os.path.join(checkpoint, 'prefix_model.bin')))
                prefix_model.to(args.device)
            result = evaluate(args, prefix_model, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    return results


if __name__ == "__main__":
    main()

