import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial
# from timm.models.vision_transformer import Attention

# from timm.layers import RmsNorm

# Taken from: https://github.com/kyungmnlee/dmf/blob/main/models/layers.py
# from .flash_attention_2_jvp import flash_attn_func as fa2_func
from .flash_attention_3_jvp import flash_attn_func as fa3_func
from jvp_flash_attention.jvp_attention import attention as jvp_attention


"""RMS Layer Normalization"""
class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Use in-place safe variant for less memory overhead
        norm_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True, dtype=torch.float32) + self.eps)
        return x * norm_x.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)
    

"""attention operator"""
def attn_op(q, k, v, op="torch_sdpa"):
    """
    op: ["default", "fa2", "fa3", "torch_sdpa", "amorehead"]
        - default: base attention implementation
        - fa2: flash attention v2 with jvp support
        - fa3: flash attention v3 with jvp support
        - torch_sdpa: PyTorch built-in scaled dot-product attention (no jvp support)
        - amorehead: https://github.com/amorehead/jvp_flash_attention
    input: q, k, v (B, L, H, D)
    output: (B, L, H, D)
    """
    if op == "fa2":
        x = fa2_func(q, k, v)
    elif op == "fa3":
        x = fa3_func(q, k, v)
    elif op == "amorehead":
        x = jvp_attention(q, k, v)
    else:
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # change to (B, H, L, D)
        if op == "torch_sdpa":
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,  # No dropout for simplicity
                is_causal=False  # Not causal for this example
            )
        elif op == "base":
            scale = q.shape[-1] ** -0.5
            q = q * scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)  #- torch.max(attn, dim=-1, keepdim=True).values.detach()
            x = attn @ v
        x = x.transpose(1, 2)
    return x


"""Custom Attention Layer"""
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_func: str = "torch_sdpa",
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.attn = partial(attn_op, op=attn_func)

    # @torch.compile
    def forward(self, x, rope=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(
            B, L, 3, self.num_heads, self.head_dim
            ).permute(2, 0, 1, 3, 4)
        # shape (B, H, L, D)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = rope(q), rope(k)
        x = self.attn(q, k, v)
        x = x.reshape(B, L, C)
        x = self.proj(x)
        return x

