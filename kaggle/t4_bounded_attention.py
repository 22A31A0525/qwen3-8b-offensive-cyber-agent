"""Qwen3/T4 memory-bounded exact causal attention fallback.

Drop-in attention backend for Transformers AttentionInterface supporting Qwen3-8B.
Does not silently remove padding/causal masking and does not shorten context.

IMPORTANT: This is a candidate fix. Tested on CPU for forward/gradient parity;
it has NOT been tested on Tesla T4 hardware in this environment.
"""

from __future__ import annotations

import os
from typing import Optional
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

BACKEND_NAME = "t4_bounded_sdpa"


def tiled_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    query_tile: int = 128,
    dropout_p: float = 0.0,
    causal: bool = True,
    checkpoint_query_tiles: bool = True,
) -> torch.Tensor:
    """Exact softmax attention calculated over Q tiles against ALL K/V.

    Input Q: [batch, query_heads, query_len, head_dim]
    K,V: [batch, kv_heads, kv_len, head_dim]
    Mask (if supplied): SDPA-style bool/additive, broadcastable [B,heads,Q,K].

    Query tiling does NOT truncate the key context and allows gradients through all
    query blocks. No inference about attention alignment for custom masks.
    """
    if query_tile < 1:
        raise ValueError("query_tile must be >=1")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Expected query/key/value shaped [B,H,T,D]")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("K and V must have same sequence length")
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("Q/K/V batch size mismatch")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Query and key head dimensions mismatch")
    if query.shape[1] % key.shape[1] != 0 or key.shape[1] != value.shape[1]:
        raise ValueError("Unsupported grouped-query attention head counts")
    if not (0.0 <= dropout_p < 1.0):
        raise ValueError("Invalid dropout probability")

    # Match Transformers' repeat_kv for Qwen3 grouped-query attention.
    groups = query.shape[1] // key.shape[1]
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    q_len, kv_len = query.shape[-2], key.shape[-2]
    if attention_mask is not None:
        if attention_mask.ndim != 4:
            raise ValueError("Expected preformatted [B,1 or H,Q,K] SDPA mask")
        if attention_mask.shape[-1] < kv_len:
            raise ValueError("Mask K dimension is too short")
        if attention_mask.shape[-2] not in (1, q_len):
            raise ValueError("Mask Q dimension is incompatible")

    chunks = []
    for start in range(0, q_len, query_tile):
        stop = min(q_len, start + query_tile)
        q_chunk = query[:, :, start:stop, :]

        if attention_mask is not None:
            # Fully preserve the existing Transformers-generated causal/padding mask.
            chunk_mask = (attention_mask if attention_mask.shape[-2] == 1
                          else attention_mask[:, :, start:stop, :])
            chunk_mask = chunk_mask[..., :kv_len]
        elif causal and q_len > 1:
            # VERY IMPORTANT: is_causal=True on a non-square tile would start
            # the triangle at row zero and corrupt causal alignment. Instead
            # construct the absolute-position causal mask for each Q slice.
            offset = kv_len - q_len  # right-aligned decode; 0 for normal training
            q_positions = torch.arange(start + offset, stop + offset, device=query.device)
            k_positions = torch.arange(kv_len, device=query.device)
            chunk_mask = (q_positions[:, None] >= k_positions[None, :])[None, None, :, :]
        else:
            # Single-token decode sees all cached past keys.
            chunk_mask = None

        # Without per-tile activation checkpointing, the math backend can retain
        # ALL tiles' softmax tensors for backward, defeating memory bounds.
        # Checkpoint each tile independently so the probabilities are recomputed
        # only while that tile's gradient is evaluated.
        def run_tile(qpart, kpart, vpart, *, local_mask=chunk_mask):
            return F.scaled_dot_product_attention(
                qpart.contiguous(), kpart, vpart,
                attn_mask=local_mask,
                dropout_p=dropout_p,
                is_causal=False,
                scale=scale,
            )

        if checkpoint_query_tiles and torch.is_grad_enabled() and (
            q_chunk.requires_grad or key.requires_grad or value.requires_grad
        ):
            y = checkpoint(run_tile, q_chunk, key, value, use_reentrant=False)
        else:
            y = run_tile(q_chunk, key, value)
        chunks.append(y)
    return torch.cat(chunks, dim=-2)


def qwen_t4_bounded_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
):
    """AttentionInterface-compatible Qwen3 attention implementation."""
    if kwargs.get("output_attentions", False):
        raise NotImplementedError("This low-memory attention returns no attention weights")
    if kwargs.get("position_bias") is not None:
        raise NotImplementedError("Qwen3-8B has no position bias; refusing to ignore one")
    if kwargs.get("sliding_window") is not None and attention_mask is None:
        raise RuntimeError("Sliding window requires an explicit mask")
    if query.shape[-2] != key.shape[-2] and query.shape[-2] > 1 and attention_mask is None:
        raise RuntimeError("Unmasked non-square prefill requires explicit verified alignment")

    tile = int(os.environ.get("T4_QUERY_TILE", "128"))
    output = tiled_scaled_dot_product_attention(
        query, key, value,
        attention_mask=attention_mask,
        scale=scaling,
        query_tile=tile,
        dropout_p=dropout,
        causal=getattr(module, "is_causal", True) if is_causal is None else is_causal,
    )
    return output.transpose(1, 2).contiguous(), None


def register_transformers_attention_backend() -> str:
    """Register BOTH attention and its mask formatter (essential for safety)."""
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask

    AttentionInterface.register(BACKEND_NAME, qwen_t4_bounded_attention)
    AttentionMaskInterface.register(BACKEND_NAME, sdpa_mask)
    return BACKEND_NAME
