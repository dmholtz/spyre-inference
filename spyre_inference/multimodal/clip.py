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

"""CLIP boundary-LayerNorm workaround for Spyre.

Only ``vision_model.pre_layrnorm``/``post_layernorm`` and
``text_model.final_layer_norm`` are swapped to ``SpyreLayerNorm``. Those three
sit at the model boundary, outside any per-block compiled graph, which is
what triggers the crash ``SpyreLayerNorm`` works around (see
``spyre_inference.custom_ops.layer_norm``). ``CLIPEncoderLayer.layer_norm1``/
``layer_norm2`` are traced inside the per-block ``torch.compile`` region
already and never hit that crashing path, so they're left as plain
``nn.LayerNorm`` -- swapping them too would be unnecessary.

Applied to the already-loaded model instance (weights included), so the
replacement ``SpyreLayerNorm`` here copies the original's already-loaded
weight/bias explicitly, rather than relying on a later ``load_weights()``
pass to populate them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm

logger = init_logger(__name__)


def patch_mm_encoder_attention() -> None:
    """Replace ``MMEncoderAttention._forward_sdpa`` with a version that calls
    ``F.scaled_dot_product_attention`` directly with a pre-built mask, bypassing
    ``torch.ops.vllm.torch_sdpa_wrapper``.

    Two problems arise when ``CLIPEncoderLayer`` is compiled with ``torch.compile``:

    1. ``torch.ops.vllm.torch_sdpa_wrapper`` is a registered custom op whose fake
       impl has a broken schema (``cu_seqlens`` is not optional), so Dynamo's shape
       inference raises ``TypeError`` when tracing the block.  The existing
       ``vit_attn.register()`` patch replaces ``apply_sdpa`` *inside* the custom op,
       but that is invisible across the custom op boundary.

    2. Even with the custom op replaced, building the attention mask inside the
       compiled block traces the CPU slice-assign into the graph.  The Spyre
       inductor backend cannot lower a ``FixedLayout('cpu', …)`` buffer.

    Fix: replace ``_forward_sdpa`` with a version that:
    - calls ``torch.ops.vllm.spyre_clip_attn_mask`` (a custom op registered in
      ``custom_ops/vit_attn.py``) to obtain the device attention mask opaquely —
      its fake impl returns the correct shape so Dynamo can trace through it
      without lowering the CPU construction, and it is fullgraph-compatible
      (unlike ``torch.compiler.disable``, which causes an illegal graph break), then
    - calls ``F.pad`` + ``F.scaled_dot_product_attention`` inline so the compute
      is fully compiled on-device.

    ``CLIPAttention.forward`` always calls ``self.attn(q, k, v)`` with no
    ``cu_seqlens``, so the CLIP vision encoder's ``_forward_sdpa`` never uses the
    chunked batching path; the replacement hardcodes ``cu_seqlens=None``.
    """
    try:
        from vllm.model_executor.layers.attention.mm_encoder_attention import (
            MMEncoderAttention,
        )
    except ImportError:
        return

    if getattr(MMEncoderAttention._forward_sdpa, "_spyre_patched", False):
        return

    from spyre_inference.multimodal.utils import STICK, align_up

    def _spyre_forward_sdpa(
        self: MMEncoderAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)
        # query/key/value: [B, S, H, D] after view_qkv_to_4d
        q = query.transpose(1, 2)  # [B, H, S, D]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        d = q.shape[-1]
        seq_pad = align_up(q_len, STICK)
        d_pad = align_up(d, STICK)
        device = q.device

        # Build (or retrieve cached) additive mask via the custom op so the
        # CPU construction is opaque to Dynamo (fullgraph-compatible).
        attn_mask = torch.ops.vllm.spyre_clip_attn_mask(q_len, bsz, seq_pad, q.dtype, device)

        pad_needed = (seq_pad, d_pad) != (q_len, d)
        if pad_needed:
            pad = (0, d_pad - d, 0, seq_pad - q_len)
            q = F.pad(q, pad)
            k = F.pad(k, pad)
            v = F.pad(v, pad)
        else:
            # torch-spyre#3770: offset operands read as offset 0; materialize.
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=self.scale
        )

        if pad_needed:
            out = out[:, :, :q_len, :d]

        out = out.transpose(1, 2)  # [B, S, H, D]

        if is_reshaped:
            out = out.reshape(bsz, q_len, -1)
        return out

    _spyre_forward_sdpa._spyre_patched = True  # type: ignore[attr-defined]
    MMEncoderAttention._forward_sdpa = _spyre_forward_sdpa  # type: ignore[method-assign]
    logger.info(
        "Spyre: replaced MMEncoderAttention._forward_sdpa to use padded SDPA "
        "with spyre_clip_attn_mask (bypasses torch.ops.vllm.torch_sdpa_wrapper "
        "and keeps CPU mask construction off-graph via custom op)."
    )


def _compile_vision_encoder_blocks(model: torch.nn.Module) -> None:
    """Compile each CLIPEncoderLayer in the vision tower in place.

    ``CLIPEncoderLayer.forward(hidden_states)`` is a pure function of its input
    (LayerNorm → MMEncoderAttention → add → LayerNorm → MLP → add) with no KV-cache,
    no mask argument, and no graph breaks -- ``fullgraph=True`` and ``dynamic=False``
    are safe. Identical layers share one ``forward`` code object, so the backend traces
    once and all layers reuse the same compiled artifact.

    Called after the boundary LayerNorms are swapped, and before
    ``_compile_for_spyre`` wraps the outer model, because the CLIP vision encoder's
    block list is filtered out by ``_is_vision_tower_path`` in
    ``_repeated_block_lists`` (vision towers run eager by default).

    No-op when the vision tower is absent or its encoder has no layers.
    """
    from spyre_inference.v1.worker import compile_guard

    vision_model = getattr(model, "vision_model", None)
    if vision_model is None:
        return
    encoder = getattr(vision_model, "encoder", None)
    if encoder is None:
        return
    layers: nn.ModuleList | None = getattr(encoder, "layers", None)
    if not layers:
        return

    seen: set[int] = set()
    for block in layers:
        if id(block) in seen:
            continue
        seen.add(id(block))
        block.compile(backend="inductor", fullgraph=True, dynamic=False)
        compile_guard.watch(block, f"{type(block).__name__} (CLIP vision block)")

    logger.info(
        "Spyre: compiled %d CLIPEncoderLayer block(s) for the vision tower "
        "(per-block, fullgraph=True, dynamic=False).",
        len(seen),
    )


def _to_spyre_layer_norm(ln: torch.nn.LayerNorm, device: torch.device) -> torch.nn.LayerNorm:
    new_ln = SpyreLayerNorm(
        list(ln.normalized_shape),
        eps=ln.eps,
        elementwise_affine=ln.elementwise_affine,
        bias=ln.bias is not None,
    ).to(device=device, dtype=ln.weight.dtype if ln.elementwise_affine else torch.float16)
    if ln.elementwise_affine:
        with torch.no_grad():
            new_ln.weight.copy_(ln.weight)
            if ln.bias is not None:
                new_ln.bias.copy_(ln.bias)
    return new_ln


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Swap CLIP's three boundary LayerNorms for ``SpyreLayerNorm`` and compile
    the vision encoder blocks in place.

    The ``isinstance`` checks are a second line of defense on top of the
    ``model_type == "clip"`` dispatch gate in ``multimodal/__init__.py``: they
    keep this a no-op (rather than an ``AttributeError`` on ``normalized_shape``)
    for any boundary norm that isn't a plain ``nn.LayerNorm``.
    """
    text_model = getattr(model, "text_model", None)
    if text_model is not None:
        ln = getattr(text_model, "final_layer_norm", None)
        if isinstance(ln, torch.nn.LayerNorm):
            text_model.final_layer_norm = _to_spyre_layer_norm(ln, device)

    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None:
        pre_ln = getattr(vision_model, "pre_layrnorm", None)
        if isinstance(pre_ln, torch.nn.LayerNorm):
            vision_model.pre_layrnorm = _to_spyre_layer_norm(pre_ln, device)
        post_ln = getattr(vision_model, "post_layernorm", None)
        if isinstance(post_ln, torch.nn.LayerNorm):
            vision_model.post_layernorm = _to_spyre_layer_norm(post_ln, device)

    logger.info_once(
        "Spyre: CLIP's boundary LayerNorms (pre_layrnorm/post_layernorm/"
        "final_layer_norm) use SpyreLayerNorm; layer_norm1/layer_norm2 inside "
        "encoder blocks are unaffected."
    )

    patch_mm_encoder_attention()
    _compile_vision_encoder_blocks(model)
