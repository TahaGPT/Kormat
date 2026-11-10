"""Own KV cache for Qwen2 (transformers 4.46 Cache interface).

Design: one pre-allocated buffer per layer, shape (B, kv_heads, max_len, head_dim), plus a single integer
``length``. ``update`` writes the new keys/values (already RoPE-rotated by the attention layer, see
modeling_qwen2.py:516-522 / :312) into ``buf[:, :, length:length+n]`` and returns a *view* ``buf[..., :length+n, :]``.

Why pre-allocated instead of torch.cat (DynamicCache, cache_utils.py:441-443): cat re-copies the whole cache every
token (O(n) bytes per step, O(n^2) overall and allocator churn), whereas a slice write is O(new tokens). It also makes
rollback for speculative decoding O(1): ``crop(n)`` only moves the integer ``length``; stale rows past ``length`` are
never read (the returned view stops at ``length``) and are overwritten by the next update.

The length is advanced after the LAST layer's update of a forward pass (all layers see the same pre-forward length).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import Cache


class PreallocKVCache(Cache):
    def __init__(self, num_layers: int, batch: int, kv_heads: int, head_dim: int, max_len: int,
                 dtype: torch.dtype = torch.float32, device="cpu"):
        super().__init__()
        self.num_layers, self.kv_heads, self.head_dim, self.max_len = num_layers, kv_heads, head_dim, max_len
        self.k = [torch.zeros(batch, kv_heads, max_len, head_dim, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v = [torch.zeros(batch, kv_heads, max_len, head_dim, dtype=dtype, device=device) for _ in range(num_layers)]
        self.length = 0            # number of valid positions (same for every layer between forwards)
        self.update_calls = 0      # instrumentation: proves the model really calls our update()
        self._pending = 0
        self.dtype = dtype

    @classmethod
    def for_model(cls, model, batch: int, max_len: int) -> "PreallocKVCache":
        c = model.config
        return cls(c.num_hidden_layers, batch, c.num_key_value_heads, c.hidden_size // c.num_attention_heads,
                   max_len, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
               cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        n = key_states.shape[-2]
        start = self.length
        if start + n > self.max_len:
            raise RuntimeError(f"KV cache overflow: {start}+{n} > {self.max_len}")
        # cache_position (from the model) must agree with our own bookkeeping
        if cache_kwargs is not None and cache_kwargs.get("cache_position") is not None:
            cp = cache_kwargs["cache_position"]
            assert int(cp[0]) == start and int(cp[-1]) == start + n - 1, (cp, start, n)
        self.k[layer_idx][:, :, start:start + n] = key_states
        self.v[layer_idx][:, :, start:start + n] = value_states
        self.update_calls += 1
        if layer_idx == self.num_layers - 1:
            self.length = start + n
        return self.k[layer_idx][:, :, :start + n], self.v[layer_idx][:, :, :start + n]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.length

    def get_max_cache_shape(self) -> Optional[int]:
        return None  # behave like a dynamic cache for mask construction (mask length = attention_mask length)

    def crop(self, max_length: int) -> None:
        """Roll back to the first ``max_length`` positions (O(1))."""
        if max_length < 0:
            max_length = self.length + max_length
        self.length = min(self.length, max_length)

    def nbytes_used(self) -> int:
        """Analytical bytes of the valid region: 2 * layers * kv_heads * head_dim * length * itemsize * batch."""
        b = self.k[0].shape[0]
        return 2 * self.num_layers * b * self.kv_heads * self.head_dim * self.length * self.k[0].element_size()
