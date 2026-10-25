# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# FreeSpec uses FlashAttention 2 explicitly so local windows are always honored.
import torch


def flash_attention(q, k, v, k_lens=None, window_size=(-1, -1)):
    """Attention over [batch, tokens, heads, channels], with optional key lengths."""
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    output_dtype = q.dtype
    dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    q, k, v = (x.to(dtype) for x in (q, k, v))
    if k_lens is None:
        result = flash_attn_func(q, k, v, window_size=window_size)
    else:
        batch, length = q.shape[:2]
        key_lengths = k_lens.to(device=q.device, dtype=torch.int32)
        cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * length
        cu_k = torch.cat([key_lengths.new_zeros(1), key_lengths]).cumsum(0, dtype=torch.int32)
        keys = torch.cat([x[:int(n)] for x, n in zip(k, k_lens)])
        values = torch.cat([x[:int(n)] for x, n in zip(v, k_lens)])
        result = flash_attn_varlen_func(
            q.flatten(0, 1), keys, values, cu_q, cu_k,
            length, int(key_lengths.max()), window_size=window_size,
        ).unflatten(0, (batch, length))
    return result.to(output_dtype)
