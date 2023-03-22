import os
import math
import logging
import json
from pathlib import Path
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM

from load import load_weights


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()

        self.divider = math.sqrt(sizes[-1] / sizes[0])
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.divider  # scaling for the initial stability
        x = self.model(x)
        return x


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=F.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=F.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=F.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):
    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int = 10,
                 clip_length: int = 10, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out


class ClipCap(nn.Module):
    def __init__(self, model_name, device, prefix_length: int = 10, clip_length: int = 40, prefix_size: int = 512,
                 num_layers: int = 1, model_path: str = '', fix_gpt: bool = False,
                 use_label_prefix: bool = False, label_path: str = '', label_length: int = 10,
                 use_transformer_mapper: bool = False, use_ptuning_v2: bool = False,
                 dropout: float = 0,
                 model_weight: str = '', scalar_output: bool = False):
        super(ClipCap, self).__init__()

        self.prefix_length = prefix_length
        self.prefix_size = prefix_size
        self.label_length = label_length
        self.scalar_output = scalar_output
        self.num_layers = num_layers
        self.use_transformer_mapper = use_transformer_mapper
        self.use_ptuning_v2 = use_ptuning_v2

        self.dropout = nn.Dropout(dropout)

        hparams = load_weights(self, AutoModelForCausalLM, model_weight, 'gpt', model_name,
                               prev_name='model')

        self.device = device
        self.gpt = self.gpt.to(self.device)

        config = self.gpt.config
        self.match_n_layer = getattr(config, 'n_layer', getattr(config, 'num_layers', None))  # gpt2 vs. gpt_neo
        self.match_n_head = getattr(config, 'n_head', getattr(config, 'num_heads', None))
        self.n_embd = getattr(config, 'n_embd', getattr(config, 'hidden_size', None))
        self.match_n_embd = self.n_embd // self.match_n_head

        self.clip_project = self.get_mapper()

        if Path(label_path).is_file():
            with open(label_path) as f:
                labels = json.load(f)
            self.labels = {i: v for v, i in labels.items()}
            if not use_label_prefix:
                log.info("adding label projections")
                self.label_project = nn.Sequential(
                    nn.Embedding(len(self.labels), self.prefix_size),
                    self.get_mapper()
                )

        if os.path.isfile(model_path):
            log.info(f"loading model from {model_path}")
            weight = torch.load(model_path, map_location=torch.device('cpu'))
            weight = {k[len('clip_project.'):]: v for k, v in weight.items()
                      if k.startswith('clip_project.')}
            self.clip_project.load_state_dict(weight)

        if fix_gpt:
            log.info("fixing gpt parameters")
            for param in self.gpt.parameters():
                param.requires_grad_(False)

        if self.scalar_output:
            self.gpt.lm_head = nn.Linear(self.gpt.transformer.embed_dim, 1).to(self.device)

        self.clip_project = self.clip_project.to(self.device)
        if hasattr(self, 'label_project'):
            self.label_project = self.label_project.to(self.device)

    def get_mapper(self):
        if self.use_ptuning_v2:
            total_embd = self.match_n_layer * 2 * self.n_embd
            module = MLP((self.prefix_size,
                          *[self.prefix_size
                            for i in range(self.num_layers)],
                          total_embd * self.prefix_length))
        elif self.use_transformer_mapper:
            log.info("using transformer mapper")
            module = TransformerMapper(self.prefix_size, self.n_embd,
                                       self.prefix_length, self.prefix_length, num_layers=self.num_layers)  # 8)
        else:
            module = MLP((self.prefix_size,
                          *[(self.n_embd * self.prefix_length) // 2
                            for i in range(self.num_layers)],
                          self.n_embd * self.prefix_length))
        return module

    def get_encoder_loss(self, input_ids: torch.Tensor, features: torch.Tensor,
                         device = None):
        input_ids = input_ids[:, :self.prefix_length].to(device)
        embedding = self.gpt.transformer.wte(input_ids)
        features = features.to(device)
        prefix_projections = self.clip_project(features.type_as(embedding)).reshape(-1, self.prefix_length, self.n_embd)
        fct = nn.MSELoss()
        loss = fct(prefix_projections, embedding.detach())
        return loss

    def forward(self, *args, **kwargs):
        if self.use_ptuning_v2:
            return self.forward_prefix(*args, **kwargs)
        else:
            return self.forward_embedding(*args, **kwargs)

    def forward_embedding(self, input_ids: torch.Tensor, features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values = None, device = None, **kwargs):

        if device is None:
            device = self.device
        input_ids = input_ids.to(device)
        if features is not None:
            features = features.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        use_labels = labels is not None and hasattr(self, 'label_project')

        embedding = self.gpt.transformer.wte(input_ids)
        embed_txt = embedding
        prefix_length = self.prefix_length
        if use_labels:
            prefix_length += self.label_length
        if past_key_values is None:
            prefix_projections = self.clip_project(features.type_as(embedding)).reshape(-1, self.prefix_length, self.n_embd)
            if use_labels:
                label_projections = self.label_project(labels.long()).reshape(-1, self.label_length, self.n_embd)
                prefix_projections = torch.cat((prefix_projections, label_projections), dim=1)
            embedding = torch.cat((prefix_projections.to(embedding.dtype), embedding), dim=1)
        if torch.is_tensor(attention_mask):
            prefix_mask = torch.ones_like(attention_mask)[:, :1].repeat(1, prefix_length)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        outputs = self.gpt(inputs_embeds=embedding, attention_mask=attention_mask,
                           past_key_values=past_key_values,
                           return_dict=True,
                           output_attentions=False,
                           output_hidden_states=True)
        if past_key_values is None:
            outputs.logits = outputs.logits[:, prefix_length:]
        return outputs

    def forward_prefix(self, input_ids: torch.Tensor, features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values = None, device = None, **kwargs):

        if device is None:
            device = self.device
        input_ids = input_ids.to(device)
        if features is not None:
            features = features.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        use_labels = labels is not None and hasattr(self, 'label_project')

        prefix_length = self.prefix_length
        if use_labels:
            prefix_length += self.label_length
        if past_key_values is None:
            prefix_projections = self.clip_project(features.type_as(self.clip_project.model[0].weight))
            prefix_projections = prefix_projections.reshape(-1, self.prefix_length,
                                                            self.match_n_layer * 2, self.match_n_head, self.match_n_embd)
            if use_labels:
                label_projections = self.label_project(labels.long())
                label_projections = label_projections.reshape(-1, self.label_length,
                                                              self.match_n_layer * 2, self.match_n_head, self.match_n_embd)
                prefix_projections = torch.cat((prefix_projections, label_projections), dim=1)
            temp_control = prefix_projections
            temp_control = self.dropout(temp_control)
            past_key_values = temp_control.permute([2, 0, 3, 1, 4]).split(2)

        if torch.is_tensor(attention_mask):
            prefix_mask = torch.ones_like(attention_mask)[:, :1].repeat(1, prefix_length)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask,
                           past_key_values=past_key_values,
                           return_dict=True,
                           output_attentions=False,
                           output_hidden_states=True)
        if past_key_values is None:
            outputs.logits = outputs.logits[:, prefix_length:]
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        features = kwargs.get("features", None)
        labels = kwargs.get("labels", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "features": features,
            "labels": labels,
        }
