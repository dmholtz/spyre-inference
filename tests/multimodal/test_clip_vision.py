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

"""Tests for CLIP vision-encoder compilation patches.

Covers:
- ``_compile_vision_encoder_blocks``: per-block torch.compile wrapping
- ``_clip_attn_mask_op``: the ``spyre_clip_attn_mask`` custom op
- ``patch_mm_encoder_attention``: ``MMEncoderAttention._forward_sdpa`` replacement

No Spyre hardware required; all tests run on CPU.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

from spyre_inference.multimodal.clip import _compile_vision_encoder_blocks
from spyre_inference.multimodal.clip import apply as apply_clip_patches
from spyre_inference.multimodal.clip import patch_mm_encoder_attention


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeEncoderLayer(nn.Module):
    """Minimal stand-in for CLIPEncoderLayer: a single Linear forward."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


def _model_with_vision_encoder(num_layers: int = 3, shared: bool = False):
    """Build a stand-in for CLIPEmbeddingModel with vision_model.encoder.layers."""
    layers = [_FakeEncoderLayer() for _ in range(num_layers)]
    if shared:
        # Simulate PP-style aliasing: every slot points at the same object.
        layers = [layers[0]] * num_layers
    encoder = types.SimpleNamespace(layers=nn.ModuleList(layers))
    vision_model = types.SimpleNamespace(encoder=encoder)
    return types.SimpleNamespace(vision_model=vision_model)


# ---------------------------------------------------------------------------
# _compile_vision_encoder_blocks
# ---------------------------------------------------------------------------


def test_compile_vision_encoder_blocks_wraps_each_layer():
    """Each layer must have been passed to block.compile() (indicated by
    ``_compiled_call_impl`` being set, which torch sets on every compiled module)."""
    model = _model_with_vision_encoder(num_layers=3)
    layers = list(model.vision_model.encoder.layers)

    _compile_vision_encoder_blocks(model)

    assert all(layer._compiled_call_impl is not None for layer in layers)


def test_compile_vision_encoder_blocks_counts_aliased_layers_once():
    """When PP aliases the same block object across multiple slots, compile() must
    be called exactly once (same semantics as _compile_blocks in the runner)."""
    model = _model_with_vision_encoder(num_layers=4, shared=True)
    layer = model.vision_model.encoder.layers[0]

    _compile_vision_encoder_blocks(model)

    assert layer._compiled_call_impl is not None


def test_compile_vision_encoder_blocks_registers_with_compile_guard(monkeypatch):
    """compile_guard.watch must be called once per unique block object."""
    from spyre_inference.v1.worker import compile_guard

    watched: list[object] = []
    monkeypatch.setattr(compile_guard, "watch", lambda target, label: watched.append(target))

    model = _model_with_vision_encoder(num_layers=3)
    unique_layers = list(dict.fromkeys(model.vision_model.encoder.layers))

    _compile_vision_encoder_blocks(model)

    assert len(watched) == len(unique_layers)
    assert all(layer in watched for layer in unique_layers)


def test_compile_vision_encoder_blocks_noop_without_vision_model():
    """A model with no vision_model attribute must not raise."""
    _compile_vision_encoder_blocks(types.SimpleNamespace())


def test_compile_vision_encoder_blocks_noop_without_encoder():
    """vision_model present but no encoder: must not raise."""
    model = types.SimpleNamespace(vision_model=types.SimpleNamespace())
    _compile_vision_encoder_blocks(model)


def test_compile_vision_encoder_blocks_noop_with_empty_layers():
    """An empty ModuleList (e.g. all layers on other PP ranks) must not raise."""
    encoder = types.SimpleNamespace(layers=nn.ModuleList([]))
    model = types.SimpleNamespace(vision_model=types.SimpleNamespace(encoder=encoder))
    _compile_vision_encoder_blocks(model)


def test_apply_compiles_vision_encoder_blocks():
    """apply() must compile the vision encoder blocks in addition to swapping the
    boundary LayerNorms -- the two are independent and both must happen."""
    model = _model_with_vision_encoder(num_layers=2)
    # apply() also touches text_model/vision_model norms; add stubs so it doesn't
    # crash on the LayerNorm swap path (the vision_model here has no pre_layrnorm).
    model.text_model = types.SimpleNamespace()

    layers = list(model.vision_model.encoder.layers)
    apply_clip_patches(model, torch.device("cpu"))

    assert all(layer._compiled_call_impl is not None for layer in layers)


# ---------------------------------------------------------------------------
# _clip_attn_mask_op (the custom op registered as spyre_clip_attn_mask)
# ---------------------------------------------------------------------------


def test_clip_attn_mask_op_shape():
    """Returned mask must be ``[b, 1, seq_pad, seq_pad]`` on the requested device."""
    from spyre_inference.custom_ops.vit_attn import _clip_attn_mask_op

    b, q_len, seq_pad = 3, 50, 64  # CLIP ViT-B/32: 50 patches, pad to 64
    mask = _clip_attn_mask_op(q_len, b, seq_pad, torch.float16, torch.device("cpu"))

    assert mask.shape == (b, 1, seq_pad, seq_pad)
    assert mask.dtype == torch.float16


def test_clip_attn_mask_op_padded_keys_are_neg_inf():
    """The ``seq_pad - q_len`` trailing key columns must be ``-inf`` so padded
    positions are never attended to."""
    from spyre_inference.custom_ops.vit_attn import _clip_attn_mask_op

    q_len, seq_pad = 50, 64
    mask = _clip_attn_mask_op(q_len, 1, seq_pad, torch.float32, torch.device("cpu"))

    real_cols = mask[0, 0, :, :q_len]
    pad_cols = mask[0, 0, :, q_len:]
    assert (real_cols == 0).all(), "real key columns should be zero (full-attend)"
    assert (pad_cols == torch.finfo(torch.float32).min).all(), (
        "padded key columns should be -inf"
    )


def test_clip_attn_mask_op_is_cached():
    """A second call with the same args must return the exact same tensor object."""
    from spyre_inference.custom_ops.vit_attn import _clip_attn_mask_op
    from spyre_inference.multimodal.utils import _MASK_ATTR, _full_attend_mask_key

    q_len, seq_pad = 50, 64
    key_tensor = _full_attend_mask_key(q_len)
    try:
        # Clear any prior cached entry so the test is independent.
        if hasattr(key_tensor, _MASK_ATTR):
            delattr(key_tensor, _MASK_ATTR)

        m1 = _clip_attn_mask_op(q_len, 2, seq_pad, torch.float16, torch.device("cpu"))
        m2 = _clip_attn_mask_op(q_len, 2, seq_pad, torch.float16, torch.device("cpu"))

        assert m1 is m2
    finally:
        if hasattr(key_tensor, _MASK_ATTR):
            delattr(key_tensor, _MASK_ATTR)


# ---------------------------------------------------------------------------
# patch_mm_encoder_attention
# ---------------------------------------------------------------------------

mm_encoder_attn = pytest.importorskip(
    "vllm.model_executor.layers.attention.mm_encoder_attention"
)
MMEncoderAttention = mm_encoder_attn.MMEncoderAttention


def _make_mm_encoder_attention(num_heads: int, head_size: int):
    """Instantiate a fake MMEncoderAttention without calling full __init__."""
    attn = object.__new__(MMEncoderAttention)
    nn.Module.__init__(attn)
    attn.num_heads = num_heads
    attn.num_kv_heads = num_heads
    attn.head_size = head_size
    attn.scale = head_size**-0.5
    return attn


@pytest.fixture
def _restore_mm_encoder_attention_forward():
    """Save and restore MMEncoderAttention._forward_sdpa around each test.

    Without restoration, the first test that calls patch_mm_encoder_attention
    poisons the class-level guard for all subsequent tests in the same process.
    """
    original = MMEncoderAttention._forward_sdpa
    try:
        yield
    finally:
        MMEncoderAttention._forward_sdpa = original


@pytest.mark.usefixtures("_restore_mm_encoder_attention_forward")
def test_patch_mm_encoder_attention_replaces_forward_sdpa():
    """After patching, ``MMEncoderAttention._forward_sdpa`` must be the Spyre
    replacement (identified by the ``_spyre_patched`` sentinel)."""
    patch_mm_encoder_attention()

    assert getattr(MMEncoderAttention._forward_sdpa, "_spyre_patched", False)


@pytest.mark.usefixtures("_restore_mm_encoder_attention_forward")
def test_patch_mm_encoder_attention_is_idempotent():
    """Calling patch_mm_encoder_attention() twice must not raise and must not
    re-assign the method (the idempotency guard must fire on the second call)."""
    patch_mm_encoder_attention()
    first = MMEncoderAttention._forward_sdpa

    patch_mm_encoder_attention()
    second = MMEncoderAttention._forward_sdpa

    assert first is second


@pytest.mark.usefixtures("_restore_mm_encoder_attention_forward")
@pytest.mark.parametrize(
    ("b", "s", "h", "d"),
    [
        (2, 64, 4, 16),  # stick-aligned seq and head-dim
        (1, 50, 8, 64),  # S=50 is the ViT-B/32 patch count (not stick-aligned)
    ],
    ids=["aligned_seq", "non_aligned_seq"],
)
def test_patch_mm_encoder_attention_forward_produces_correct_shape(b, s, h, d):
    """The patched ``_forward_sdpa`` must return ``[B, S, H*D]`` from a
    ``[B, S, H*D]`` 3-D input (the ``is_reshaped`` path), across both aligned and
    non-aligned sequence lengths."""
    patch_mm_encoder_attention()

    hidden = h * d
    attn = _make_mm_encoder_attention(num_heads=h, head_size=d)

    out = attn._forward_sdpa(
        torch.randn(b, s, hidden),
        torch.randn(b, s, hidden),
        torch.randn(b, s, hidden),
    )

    assert out.shape == (b, s, hidden)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
