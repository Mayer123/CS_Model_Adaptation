import torch
import torch.nn as nn 
from transformers.modeling_outputs import Seq2SeqLMOutput, CausalLMOutputWithCrossAttentions
from typing import *

class PrefixModelBart(nn.Module):
    def __init__(self, prefix_len, config, mid_dim=512, device=None) -> None:
        super(PrefixModelBart, self).__init__()

        self.prefix_len = prefix_len
        self.prefix = torch.arange(prefix_len).long().to(device)
        self.n_layers = config.encoder_layers 
        self.kv = 2 
        self.num_heads = config.encoder_attention_heads

        self.enc_prefix_emb = nn.Embedding(prefix_len, config.d_model)
        self.enc_trans = nn.Sequential(
            nn.Linear(config.d_model, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, self.n_layers * self.kv * config.d_model))
        
        self.dec_prefix_emb = nn.Embedding(prefix_len, config.d_model)
        self.dec_trans = nn.Sequential(
            nn.Linear(config.d_model, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, self.n_layers * self.kv * config.d_model))

        self.enc_dec_prefix_emb = nn.Embedding(prefix_len, config.d_model)
        self.enc_dec_trans = nn.Sequential(
            nn.Linear(config.d_model, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, self.n_layers * self.kv * config.d_model))
           
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, model, input_ids, attention_mask, decoder_input_ids, labels=None, use_cache=None):        
        all_prefix = self.get_prefix(input_ids)
        outputs = model(input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            prefix_key_values=all_prefix,
            use_cache=use_cache,
        )
        if labels is None:
            return outputs
        masked_lm_loss = self.loss_fct(outputs[0].view(-1, model.config.vocab_size), labels.view(-1))
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=outputs[0],
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def get_prefix(self, input_ids):
        bsz, seq_len = input_ids.shape
        input_prefix = self.prefix.unsqueeze(0).expand(bsz, -1)
        
        enc_prefix = self.enc_prefix_emb(input_prefix)
        enc_prefix = self.enc_trans(enc_prefix)  
        enc_prefix = enc_prefix.view(bsz, self.prefix_len, self.n_layers * self.kv, self.num_heads, -1)
        enc_prefix = enc_prefix.permute([2, 0, 3, 1, 4]).split(self.kv)

        dec_prefix = self.dec_prefix_emb(input_prefix)
        dec_prefix = self.dec_trans(dec_prefix)    
        dec_prefix = dec_prefix.view(bsz, self.prefix_len, self.n_layers * self.kv, self.num_heads, -1)
        dec_prefix = dec_prefix.permute([2, 0, 3, 1, 4]).split(self.kv)

        enc_dec_prefix = self.enc_dec_prefix_emb(input_prefix)
        enc_dec_prefix = self.enc_dec_trans(enc_dec_prefix) 
        enc_dec_prefix = enc_dec_prefix.view(bsz, self.prefix_len, self.n_layers * self.kv, self.num_heads, -1)
        enc_dec_prefix = enc_dec_prefix.permute([2, 0, 3, 1, 4]).split(self.kv)

        result = []
        for i, key_val in enumerate(enc_prefix):
            this_layer = {'encoder': [key_val[0].contiguous(), key_val[1].contiguous(), torch.zeros(bsz, 1, seq_len, self.prefix_len).to(key_val.device)]}
            this_layer['self'] = [dec_prefix[i][0].contiguous(), dec_prefix[i][1].contiguous(), torch.zeros(bsz, 1, 1, self.prefix_len).to(key_val.device)]
            this_layer['encoder_decoder'] = [enc_dec_prefix[i][0].contiguous(), enc_dec_prefix[i][1].contiguous(), torch.zeros(bsz, 1, 1, self.prefix_len).to(key_val.device)]
            result.append(this_layer)
        return result

class PrefixModelGPT2(nn.Module):
    def __init__(self, prefix_len, config, mid_dim=512, device=None) -> None:
        super(PrefixModelGPT2, self).__init__()

        self.prefix_len = prefix_len
        self.prefix = torch.arange(prefix_len).long().to(device)
        self.n_layers = config.n_layer
        self.kv = 2 
        self.num_heads = config.n_head
        
        self.dec_prefix_emb = nn.Embedding(prefix_len, config.n_embd)
        self.dec_trans = nn.Sequential(
            nn.Linear(config.n_embd, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, self.n_layers * self.kv * config.n_embd))
           
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, model, input_ids, labels=None, use_cache=None):        
        all_prefix = self.get_prefix(input_ids)
        outputs = model(input_ids=input_ids,
            prefix_key_values=all_prefix,
            use_cache=use_cache,
        )
        if labels is None:
            return outputs
        loss = self.loss_fct(outputs[0].view(-1, model.config.vocab_size), labels.view(-1))
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def get_prefix(self, input_ids):
        bsz, seq_len = input_ids.shape
        input_prefix = self.prefix.unsqueeze(0).expand(bsz, -1)
        
        dec_prefix = self.dec_prefix_emb(input_prefix)
        dec_prefix = self.dec_trans(dec_prefix)    
        dec_prefix = dec_prefix.view(bsz, self.prefix_len, self.n_layers * self.kv, self.num_heads, -1)
        dec_prefix = dec_prefix.permute([2, 0, 3, 1, 4]).split(self.kv)

        result = []
        for i, key_val in enumerate(dec_prefix):
            this_layer = {'self': [dec_prefix[i][0].contiguous(), dec_prefix[i][1].contiguous()]}
            result.append(this_layer)
        return result
