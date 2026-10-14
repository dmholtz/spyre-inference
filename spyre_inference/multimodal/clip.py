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

"""CLIP workarounds for Spyre: boundary LayerNorms and vision-tower compile.

All five LayerNorms -- ``pre_layrnorm``, ``post_layernorm``, ``final_layer_norm``
and every ``CLIPEncoderLayer.layer_norm1``/``layer_norm2`` -- are swapped to
``SpyreLayerNorm``.

**Why LayerNorm**: ``nn.LayerNorm`` dispatches through ``aten.layer_norm.default``,
which torch-spyre decomposes via an ``_OPWrapper`` that unconditionally compiles
``spyre_layer_norm`` the first time it runs eagerly (regardless of
``--enforce-eager``).  For a second request with a different image-batch size the
OPWrapper's Inductor graph recompiles, and the layout-propagation pass fails::

    Multi-arg pointwise with mixed EA: STANDARD input arg0_1 must broadcast …
    Its stick maps to host dim 0 of size 768

The conflict is the staggered fp32 intermediate inside ``spyre_layer_norm``'s
decomposition vs. the fp16 ``weight`` parameter (STANDARD layout).
``SpyreLayerNorm`` avoids this by implementing normalisation with plain fp32
arithmetic (casting ``weight``/``bias`` to fp32 too) so both sides always share
the same EA.

**Why residual adds**: after the norm fix, the residual adds in each
``CLIPEncoderLayer`` (``residual + attn_out`` and ``residual + mlp_out``) still
crash with the same mixed-EA error in eager mode because the two operands come
from separate OPWrapper graphs.  ``_patched_encoder_layer_forward`` replaces each
add with ``layer_norm.fused_add(residual, out)``, which compiles ``a + b`` with
``force=True`` so the layout pass sees both operands in one graph.

Applied to the already-loaded model instance (weights included), so the
replacement ``SpyreLayerNorm`` here copies the original's already-loaded
weight/bias explicitly, rather than relying on a later ``load_weights()``
pass to populate them.

The vision tower's encoder layers are marked for the runner's per-block compile. Their
body builds nothing on the host once ``MMEncoderAttention`` is Spyre's OOT
implementation (``custom_ops/vit_attn.py``), so ``patch_vision_tower`` checks that
instead of assuming it. It takes the tower itself, not the model, so a model that
embeds a CLIP tower can reuse it.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm
from spyre_inference.multimodal.utils import mark_blocks_compilable

logger = init_logger(__name__)


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


def _patched_encoder_layer_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """`CLIPEncoderLayer.forward` with fused residual adds via ``SpyreLayerNorm.fused_add``.

    Mirrors the original exactly, but replaces each ``residual + out`` add with
    ``layer_norm.fused_add(residual, out)``.  In eager mode the residual and the
    attention/MLP output come from different OPWrapper compiled graphs whose Spyre
    EAs are incompatible; ``fused_add`` compiles ``a + b`` with ``force=True`` so
    the layout-propagation pass sees both operands in one graph and assigns them
    compatible EAs.
    """
    # attention sub-block
    residual = hidden_states
    hidden_states, _ = self.self_attn(hidden_states=self.layer_norm1(hidden_states))
    hidden_states = self.layer_norm1.fused_add(residual, hidden_states)

    # feedforward sub-block
    residual = hidden_states
    hidden_states = self.mlp(self.layer_norm2(hidden_states))
    hidden_states = self.layer_norm2.fused_add(residual, hidden_states)

    return hidden_states


def patch_vision_tower(tower: torch.nn.Module, device: torch.device) -> None:
    """Swap ``CLIPVisionTransformer`` LayerNorms and mark its encoder layers compilable."""
    pre_ln = getattr(tower, "pre_layrnorm", None)
    if isinstance(pre_ln, torch.nn.LayerNorm):
        tower.pre_layrnorm = _to_spyre_layer_norm(pre_ln, device)
    post_ln = getattr(tower, "post_layernorm", None)
    if isinstance(post_ln, torch.nn.LayerNorm):
        tower.post_layernorm = _to_spyre_layer_norm(post_ln, device)

    layers = getattr(getattr(tower, "encoder", None), "layers", None)
    if not layers:
        return

    # Swap layer_norm1/layer_norm2 and patch forward on every encoder block.
    # SpyreLayerNorm.fused_add compiles the residual add with force=True so
    # the layout pass sees both operands in one graph.
    import types

    for layer in layers:
        for attr in ("layer_norm1", "layer_norm2"):
            ln = getattr(layer, attr, None)
            if isinstance(ln, torch.nn.LayerNorm):
                setattr(layer, attr, _to_spyre_layer_norm(ln, device))
        layer.forward = types.MethodType(_patched_encoder_layer_forward, layer)

    # Deferred: vit_attn imports this package's utils, so a top-level import is a cycle.
    from spyre_inference.custom_ops.vit_attn import SpyreMMEncoderAttention

    # A layer whose attention is not the OOT impl would trace upstream's opaque SDPA
    # wrapper, which pads nothing; leave such a tower eager.
    attns = [getattr(getattr(layer, "self_attn", None), "attn", None) for layer in layers]
    if all(isinstance(attn, SpyreMMEncoderAttention) for attn in attns):
        mark_blocks_compilable(layers)
    else:
        logger.warning_once(
            "Spyre: CLIP vision layers do not use SpyreMMEncoderAttention; the tower stays eager."
        )


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Swap CLIP's three boundary LayerNorms for ``SpyreLayerNorm``, in place, and
    mark the vision tower's encoder layers compilable.

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
        patch_vision_tower(vision_model, device)

    logger.info_once(
        "Spyre: all CLIP LayerNorms use SpyreLayerNorm; encoder block forwards "
        "patched to use fused_add for residual adds."
    )
