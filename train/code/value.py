from typing import Optional

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# from model.gpt2 import GPT2ForTokenRegression
from utils.utils import mask_pad

from clipcap import ClipCap


class Value(nn.Module):
    def __init__(self, model_name, device, clipcap_path='', fix_gpt=False,
                 use_transformer_mapper: bool = False, use_ptuning_v2: bool = False,
                 prefix_length=10, clipcap_num_layers: int = 1,
                 unfix_value_model: bool = False,
                 label_path: str = '', model_weight: str = '', use_label_prefix: bool = False):
        super().__init__()

        self.device = device

        fix_gpt = fix_gpt and not unfix_value_model
        self.model = ClipCap(model_name, device,
                             model_path=clipcap_path, fix_gpt=fix_gpt,
                             prefix_length=prefix_length,
                             num_layers=clipcap_num_layers,
                             label_path=label_path, model_weight=model_weight,
                             use_transformer_mapper=use_transformer_mapper,
                             use_ptuning_v2=use_ptuning_v2,
                             use_label_prefix=use_label_prefix, scalar_output=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="<|endoftext|>")
        self.model.gpt.config.pad_token_id = self.tokenizer.pad_token_id

    def forward_pass(self,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor,
                     features: torch.Tensor,
                     labels: Optional[torch.Tensor],
                     device = None):

        if device is None:
            device = self.device

        batch_size, query_seq_len = query_input_ids.shape
        input_ids = torch.cat([query_input_ids, response_input_ids], dim=-1)
        attention_mask = torch.cat([query_mask, response_mask], dim=-1)

        # forward pass to get next token
        outputs = self.model(
            input_ids,
            features,
            attention_mask,
            labels,
            device=device
        )
        # get the second to last logit
        logits = outputs.logits.squeeze(-1)
        logits = logits[:, query_seq_len-1:-1]

        return {
            'response/value': mask_pad(logits, response_mask)
        }
