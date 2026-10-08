# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers shared by the vision-tower workarounds in this package."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from spyre_inference.custom_ops.utils import convert

# Spyre stick width in 2-byte elements (128-byte stick). Matmul reduction dims and
# the sequence axis must land on it.
STICK = 64


def align_up(n: int, align: int = STICK) -> int:
    return (n + align - 1) // align * align


# Attribute names under which masks are cached on the key tensor returned by
# ``_full_attend_mask_key``. Each user gets its own attribute to avoid collisions.
_MASK_ATTR = "_spyre_padded_mask"
_VISION_MASK_ATTR = "_spyre_vision_attn_mask"


@lru_cache(maxsize=16)
def _full_attend_mask_key(seq: int) -> torch.Tensor:
    """Stable per-length tensor used as a cache handle by mask builders.

    The same object is always returned for the same ``seq``, so callers can
    attach cached masks via ``getattr``/``setattr`` without a separate dict.
    Never moved to a device.
    """
    return torch.ones(seq, seq, dtype=torch.bool)


def _padded_attn_mask(
    mask: torch.Tensor,
    b: int,
    seq: int,
    seq_pad: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive ``[b, 1, seq_pad, seq_pad]`` mask on ``device``, cached on ``mask``."""
    key = (b, seq, seq_pad, dtype, str(device))
    cached = getattr(mask, _MASK_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]

    # Assembled on CPU: strided slice-assign is not stick-safe on-device.
    neg_inf = torch.finfo(dtype).min
    m = torch.zeros(b, 1, seq_pad, seq_pad, dtype=dtype)
    m[:, :, :, seq:] = neg_inf  # padded keys never attended
    mc = convert(mask, "cpu")
    if mc.dtype == torch.bool:
        m[:, :, :seq, :seq] = torch.zeros(seq, seq, dtype=dtype).masked_fill(
            ~mc.reshape(seq, seq), neg_inf
        )
    else:
        m[:, :, :seq, :seq] = mc.to(dtype).reshape(seq, seq)

    m = convert(m, device)
    setattr(mask, _MASK_ATTR, (key, m))
    return m


def padded_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """SDPA over ``[B, H, L, D]`` with L and D stick-aligned, then cropped.

    Padding is a correctness requirement: sequence lengths not divisible by the
    stick produce wrong results on-device. Padded keys are masked to ``-inf``;
    padded queries are cropped after the call.

    ``scale`` defaults to the unpadded head dim. Pass it explicitly when the
    model carries its own scale.
    """
    b, _, seq, d = q.shape
    if scale is None:
        scale = d**-0.5
    seq_pad = align_up(seq)
    d_pad = align_up(d)
    device = q.device
    padded = (seq_pad, d_pad) != (seq, d)

    if padded:
        # F.pad tuple is last-dim-first: (D_left, D_right, L_left, L_right).
        pad = (0, d_pad - d, 0, seq_pad - seq)
        q = F.pad(q, pad)
        k = F.pad(k, pad)
        v = F.pad(v, pad)
    else:
        # Offset operands must be materialized (torch-spyre#3770).
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=_padded_attn_mask(mask, b, seq, seq_pad, q.dtype, device),
        scale=scale,
        enable_gqa=enable_gqa,
    )

    if padded:
        out = out[:, :, :seq, :d]
    return out
