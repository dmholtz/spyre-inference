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

**Why**: ``nn.LayerNorm`` dispatches through ``aten.layer_norm.default``, which
torch-spyre decomposes via an ``_OPWrapper`` that unconditionally compiles
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

import functools

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

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


def _make_compiled_forward(layer: torch.nn.Module) -> None:
    """Attach a ``force=True`` compiled forward to an encoder layer.

    In eager mode (``CompilationMode.NONE``) each op in the encoder layer is
    compiled by torch-spyre's per-op OPWrapper as a separate tiny Inductor graph.
    The residual adds (``residual + attn_out`` and ``residual + mlp_out``) then
    receive one STANDARD-layout tensor and one staggered-EA tensor from separate
    OPWrapper graphs, and the layout-propagation pass fails:

        Multi-arg pointwise with mixed EA: STANDARD input arg1_1 must broadcast …

    Compiling the entire ``forward`` with ``force=True`` puts every op -- the norm,
    attention, and residual add -- into a single Inductor graph, so the layout pass
    sees both operands of the add together and can assign compatible EAs.

    Uses a per-layer ``_spyre_compiled_forward`` slot (not ``CompileOutermost``'s
    shared ``spyre_compiled_kernel``) so it does not interfere with any other
    ``CompileOutermost`` kernel on the layer.

    In compiled block mode the layer's forward is already compiled by the runner's
    ``block.compile()`` call; ``torch.compiler.is_compiling()`` is True inside that
    graph, so this wrapper inlines transparently.
    """
    from spyre_inference.v1.worker import compile_guard

    original_forward = layer.forward
    layer._spyre_compiled_forward = None

    @functools.wraps(original_forward)
    def patched_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            # Already inside a compiled graph (block compile or outer trace) --
            # call the original forward directly so it inlines into that graph.
            return original_forward(hidden_states)
        compiled = layer._spyre_compiled_forward
        if compiled is None:
            compile_guard.watch(
                original_forward,
                f"{type(layer).__name__}.forward (encoder block)",
            )
            compiled = torch.compile(
                original_forward,
                backend=current_platform.simple_compile_backend,
                fullgraph=True,
                dynamic=False,
            )
            layer._spyre_compiled_forward = compiled
        return compiled(hidden_states)

    # Assign directly, not via types.MethodType: original_forward is already a bound
    # method (closes over `layer`), and patched_forward closes over original_forward.
    # Binding again via MethodType would prepend `layer` as an extra positional arg.
    layer.forward = patched_forward


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

    for layer in layers:
        for attr in ("layer_norm1", "layer_norm2"):
            ln = getattr(layer, attr, None)
            if isinstance(ln, torch.nn.LayerNorm):
                setattr(layer, attr, _to_spyre_layer_norm(ln, device))
        _make_compiled_forward(layer)

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
    """Swap CLIP's boundary LayerNorms for ``SpyreLayerNorm``, in place, and
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

    logger.info_once("Spyre: all CLIP LayerNorms use SpyreLayerNorm.")
