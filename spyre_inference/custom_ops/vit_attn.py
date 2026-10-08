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

"""Stick-aligned SDPA for vLLM's generic ViT attention path, and the
``spyre_clip_attn_mask`` custom op for CLIP's per-block compiled attention.

``vllm.v1.attention.ops.vit_attn_wrappers.apply_sdpa`` (used by
``MMEncoderAttention``, the default ViT attention for models like CLIP that don't
define a bespoke vision Attention) calls ``F.scaled_dot_product_attention``
directly on whatever sequence length the image produces. torch-spyre compiles
that op internally on every dispatch (regardless of ``--enforce-eager``) and its
BMM-padding pass asserts when the sequence length isn't a multiple of the
64-element fp16 stick (e.g. CLIP ViT-B/32's 50 patches). Routes through the same
``padded_sdpa`` helper Pixtral's vision tower uses (``multimodal/utils.py``),
with an "attend everywhere" mask since this path has no real one of its own.

``spyre_clip_attn_mask`` is a separate custom op used by the compiled
``CLIPEncoderLayer`` blocks (``multimodal/clip.py``). The attention mask is a
static ``[b, 1, seq_pad, seq_pad]`` tensor whose content depends only on
scalar shapes, not on tensor data. Building it inside a ``fullgraph=True``
compiled block traces the CPU slice-assign into the graph, which the Spyre
inductor backend cannot lower. Registering it as a custom op with a shape-only
fake impl makes it opaque to Dynamo while remaining fullgraph-compatible
(unlike ``torch.compiler.disable``, which causes an illegal graph break).
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from spyre_inference.multimodal.utils import (
    _VISION_MASK_ATTR,
    _full_attend_mask_key,
    padded_sdpa,
)

logger = init_logger(__name__)


# Alias: ``padded_sdpa`` caches on the mask object's identity, so we need a stable
# per-length tensor. ``_full_attend_mask_key`` from utils already provides exactly
# that (same lru_cache contract, same dtype) — no need for a second copy.
_full_attend_mask = _full_attend_mask_key


def _padded_apply_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Drop-in replacement for ``vit_attn_wrappers.apply_sdpa``.

    Input/output shape: ``(batch, seq, num_heads, head_size)``.
    """
    seq = q.shape[1]
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))  # -> (batch, heads, seq, head_size)
    out = padded_sdpa(q, k, v, _full_attend_mask(seq), scale=scale, enable_gqa=enable_gqa)
    return out.transpose(1, 2)


def _clip_attn_mask_op(
    q_len: int,
    b: int,
    seq_pad: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Real impl: build (and cache) the ``[b, 1, seq_pad, seq_pad]`` additive mask.

    Called at runtime (outside the compiled graph) for each distinct
    ``(q_len, b, seq_pad, dtype, device)`` combination. Results are cached on
    the stable key object from ``_full_attend_mask_key`` so mask construction
    happens at most once per image shape per device.
    """
    from spyre_inference.custom_ops.utils import convert

    mask = _full_attend_mask_key(q_len)
    key = (b, q_len, seq_pad, dtype, str(device))
    cached = getattr(mask, _VISION_MASK_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]

    neg_inf = torch.finfo(dtype).min
    m = torch.zeros(b, 1, seq_pad, seq_pad, dtype=dtype)
    m[:, :, :, q_len:] = neg_inf  # padded key positions are never attended
    m = convert(m, device)
    setattr(mask, _VISION_MASK_ATTR, (key, m))
    return m


def _clip_attn_mask_fake(
    q_len: int,
    b: int,
    seq_pad: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Fake impl for Dynamo shape inference: return a correctly-shaped empty tensor."""
    return torch.empty(b, 1, seq_pad, seq_pad, dtype=dtype, device=device)


def _ensure_clip_attn_mask_registered() -> None:
    """Register the ``spyre_clip_attn_mask`` custom op if not already done.

    Idempotent: safe to call from both ``register()`` and
    ``patch_mm_encoder_attention()`` so the op is always available before the
    first compiled forward, regardless of call order.
    """
    if hasattr(torch.ops.vllm, "spyre_clip_attn_mask"):
        return
    direct_register_custom_op(
        op_name="spyre_clip_attn_mask",
        op_func=_clip_attn_mask_op,
        fake_impl=_clip_attn_mask_fake,
        dispatch_key="CompositeExplicitAutograd",
    )
    logger.debug_once("Registered custom op: spyre_clip_attn_mask")


def register() -> None:
    import vllm.v1.attention.ops.vit_attn_wrappers as vit_attn_wrappers

    if getattr(vit_attn_wrappers.apply_sdpa, "_spyre_patched", False):
        return

    _padded_apply_sdpa._spyre_patched = True
    vit_attn_wrappers.apply_sdpa = _padded_apply_sdpa  # ty: ignore[invalid-assignment]
    logger.debug_once(
        "Patched vllm.v1.attention.ops.vit_attn_wrappers.apply_sdpa to pad to "
        "the 64-element stick before calling SDPA."
    )

    _ensure_clip_attn_mask_registered()
