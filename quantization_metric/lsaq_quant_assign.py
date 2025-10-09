import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import json
import math
import torch.nn.functional as F


from datasets import load_dataset

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

class W8A16Linear(nn.Module):
    def __init__(
        self,
        # bit_width,
        in_features,
        out_features,
        bias=True,
        quantize_output=False,
    ):
        super().__init__()
        # self.bit_width = bit_width
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

    def to(self, *args, **kwargs):
        super(W8A16Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        y = torch.functional.F.linear(x, self.weight, self.bias)
        return y

    @staticmethod
    def from_float(
        bit, module, weight_quant="per_channel", quantize_output=False
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A16Linear(
            # bit,
            module.in_features,
            module.out_features,
            module.bias is not None,
            quantize_output=quantize_output,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(module.weight, bit)
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(module.weight, bit)
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A16Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name})"

def quantize_llama_like(
    model, mlp_quant, self_attn_quant, low_bit, weight_quant="per_channel", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, LlamaMLP):
            if low_bit == 0:
                continue
            else:
                if name in mlp_quant:
                    bit = low_bit
                    print(f'{name} {bit} bit quant ')
                else:
                    continue
                    # if low_bit == 4:
                    #     bit = 8
                    #     print(f'{name} {bit} bit quant ')
                    # elif low_bit == 8:
                    #     continue

            m.gate_proj = W8A16Linear.from_float(
                bit, m.gate_proj, weight_quant=weight_quant
            )
            m.up_proj = W8A16Linear.from_float(
                bit, m.up_proj, weight_quant=weight_quant
            )
            m.down_proj = W8A16Linear.from_float(
                bit, m.down_proj, weight_quant=weight_quant
            )
        elif isinstance(m, LlamaAttention):
            if low_bit == 0:
                continue
            else:
                if name in self_attn_quant:
                    bit = low_bit
                    print(f'{name} {bit} bit quant ')
                else:
                    continue
                    # if low_bit == 4:
                    #     bit = 8
                    # elif low_bit == 8:
                    #     continue

            m.q_proj = W8A16Linear.from_float(
                bit,  
                m.q_proj,
                weight_quant=weight_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A16Linear.from_float(
                bit, 
                m.k_proj,
                weight_quant=weight_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A16Linear.from_float(
                bit, 
                m.v_proj,
                weight_quant=weight_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W8A16Linear.from_float(
                bit, m.o_proj, weight_quant=weight_quant
            )
            
    return model