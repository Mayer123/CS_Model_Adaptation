import json
import linecache
import os
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import torch
from torch.utils.data import Dataset

from transformers import BartTokenizer, GPT2Tokenizer

logger = getLogger(__name__)

class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
    ):
        super().__init__()
        self.type_path = type_path
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        if isinstance(self.tokenizer, BartTokenizer):
            source_inputs = self.tokenizer(source_line+' '+self.tokenizer.mask_token, add_prefix_space=True)
        elif isinstance(self.tokenizer, GPT2Tokenizer):
            source_inputs = self.tokenizer(source_line, add_prefix_space=True)
        else:
            raise ValueError("Tokenizer not recognized")
        target_inputs = self.tokenizer(source_line+' '+tgt_line, add_prefix_space=True)

        source_ids = source_inputs["input_ids"]
        target_ids = target_inputs["input_ids"]
        src_mask = source_inputs["attention_mask"]
        qid = self.type_path+str(index)
        return {
            "qid": qid,
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_src_len = max([len(x['input_ids']) for x in batch])
        max_tgt_len = max([len(x['decoder_input_ids']) for x in batch])
        input_ids = []
        masks = []
        target_ids = []
        for x in batch:
            input_ids.append(x['input_ids']+[pad_token_id]*(max_src_len-len(x['input_ids'])))
            masks.append(x['attention_mask']+[0]*(max_src_len-len(x['attention_mask'])))
            target_ids.append(x['decoder_input_ids']+[pad_token_id]*(max_tgt_len-len(x['decoder_input_ids'])))
        source_ids = torch.tensor(input_ids, dtype=torch.long)
        source_mask = torch.tensor(masks, dtype=torch.long)
        y = torch.tensor(target_ids, dtype=torch.long)
        qids = [x['qid'] for x in batch]
        batch = {
            "qids": qids,
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
        }
        return batch

class TriggerDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length=512,
        max_target_length=512,
        type_path="train",
        n_obs=None,
        prefix="",
        model_type=None,
    ):
        super().__init__()
        self.type_path = type_path
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.model_type = model_type
        if not hasattr(tokenizer, 'trigger_token'):
            raise ValueError(
                'Tokenizer missing special trigger tokens in vocab.'
                'Use `utils.add_special_tokens` to add them.'
            )

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        if isinstance(self.tokenizer, BartTokenizer):
            source_inputs = self.tokenizer(source_line+' '+self.tokenizer.mask_token, add_prefix_space=True)
        elif isinstance(self.tokenizer, GPT2Tokenizer):
            source_inputs = self.tokenizer(source_line, add_prefix_space=True)
        else:
            raise ValueError("Tokenizer not recognized")
        target_inputs = self.tokenizer(source_line+' '+tgt_line, add_prefix_space=True)

        source_ids = source_inputs["input_ids"]
        target_ids = target_inputs["input_ids"]
        src_mask = source_inputs["attention_mask"]
        qid = self.type_path+str(index)
        return {
            "qid": qid,
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_src_len = max([len(x['input_ids']) for x in batch])
        max_tgt_len = max([len(x['decoder_input_ids']) for x in batch])
        input_ids = []
        masks = []
        target_ids = []
        for x in batch:
            input_ids.append(x['input_ids']+[pad_token_id]*(max_src_len-len(x['input_ids'])))
            masks.append(x['attention_mask']+[0]*(max_src_len-len(x['attention_mask'])))
            target_ids.append(x['decoder_input_ids']+[pad_token_id]*(max_tgt_len-len(x['decoder_input_ids'])))
        source_ids = torch.tensor(input_ids, dtype=torch.long)
        source_mask = torch.tensor(masks, dtype=torch.long)
        y = torch.tensor(target_ids, dtype=torch.long)
        qids = [x['qid'] for x in batch]

        if self.model_type == 'bart':
            decoder_input_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            for i in range(len(lm_labels)):
                for j in range(len(lm_labels[i])):
                    if lm_labels[i][j] == source_ids[i][j+1]:
                        lm_labels[i][j] = -100
                    else:
                        break
            lm_labels[lm_labels == pad_token_id] = -100
        elif self.model_type == 'gpt2':
            decoder_input_ids = y[:, :-1].contiguous()
            lm_labels = y.clone()[:, 1:].clone()
            for i in range(len(lm_labels)):
                for j in range(1, len(source_ids[i])):
                    if lm_labels[i][j-1] == source_ids[i][j]:
                        lm_labels[i][j-1] = -100
                    else:
                        break
            lm_labels[lm_labels == pad_token_id] = -100

        decoder_trigger_mask = decoder_input_ids.eq(self.tokenizer.trigger_token_id)
        trigger_mask = source_ids.eq(self.tokenizer.trigger_token_id)

        batch = {
            "qids": qids,
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": lm_labels,
            "trigger_mask": trigger_mask, 
            "decoder_trigger_mask": decoder_trigger_mask,
        }
        return batch


class CommonGenDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length=512,
        max_target_length=512,
        type_path="train",
        n_obs=None,
        prefix="",
    ):
        super().__init__()
        self.type_path = type_path
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = self.prefix + linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
       
        source_inputs = self.tokenizer(source_line, add_prefix_space=True)
        target_inputs = self.tokenizer(tgt_line, add_prefix_space=True)

        source_ids = source_inputs["input_ids"]
        target_ids = target_inputs["input_ids"]
        src_mask = source_inputs["attention_mask"]
        qid = self.type_path+str(index)
        return {
            "qid": qid,
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_src_len = max([len(x['input_ids']) for x in batch])
        max_tgt_len = max([len(x['decoder_input_ids']) for x in batch])
        input_ids = []
        masks = []
        target_ids = []
        for x in batch:
            input_ids.append(x['input_ids']+[pad_token_id]*(max_src_len-len(x['input_ids'])))
            masks.append(x['attention_mask']+[0]*(max_src_len-len(x['attention_mask'])))
            target_ids.append(x['decoder_input_ids']+[pad_token_id]*(max_tgt_len-len(x['decoder_input_ids'])))
        source_ids = torch.tensor(input_ids, dtype=torch.long)
        source_mask = torch.tensor(masks, dtype=torch.long)
        y = torch.tensor(target_ids, dtype=torch.long)
        qids = [x['qid'] for x in batch]
        if self.prefix == '':
            batch = {
                "qids": qids,
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "decoder_input_ids": y,
            }
            return batch
        else:
            decoder_input_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[lm_labels == pad_token_id] = -100
            lm_labels[lm_labels == self.tokenizer.trigger_token_id] = -100
            decoder_trigger_mask = decoder_input_ids.eq(self.tokenizer.trigger_token_id)
            trigger_mask = source_ids.eq(self.tokenizer.trigger_token_id)

            batch = {
                "qids": qids,
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "decoder_input_ids": decoder_input_ids,
                "labels": lm_labels,
                "trigger_mask": trigger_mask, 
                "decoder_trigger_mask": decoder_trigger_mask,
            }
            return batch