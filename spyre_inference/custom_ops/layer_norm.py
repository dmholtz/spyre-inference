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

"""Spyre-safe ``torch.nn.LayerNorm`` replacement.

``torch.nn.LayerNorm`` is plain PyTorch, not a vLLM ``CustomOp`` (unlike
``RMSNorm``), so it can't be intercepted via ``register_oot``. Its default
forward goes through ``aten.layer_norm.default``, which torch-spyre decomposes
into three native ops (``exx2`` -> ``layernormscale`` -> ``layernormnorm``,
see ``torch_spyre._inductor.decompositions.spyre_layer_norm``). That
decomposition is wrapped in an ``_OPWrapper`` that unconditionally
``torch.compile``s it in isolation the first time it's dispatched eagerly --
regardless of ``--enforce-eager``. For call sites outside any per-block
compiled region (e.g. CLIP's ``pre_layrnorm``/``post_layernorm``), the
resulting fused kernel fails at the native ``dxp_standalone`` compiler stage
("Not enough dimensions" matching ``layernormnorm.ddl``'s fixed 5-dim
signature).

Fix: ``SpyreLayerNorm`` reimplements the forward pass with plain
mean/var/rsqrt arithmetic, so ``aten.layer_norm.default`` (and its crashing
decomposition) is never invoked at all. The mean/var reduction accumulates in
fp32 -- unlike fp16, which overflows to ``inf`` once ``|x - mean| > 256`` and
silently zeroes the row via ``rsqrt(inf)`` -- matching what ``F.layer_norm``
itself does internally, and needed here because CLIP's boundary norms are
exactly where activation outliers of that magnitude show up.

``SpyreLayerNorm`` also exposes a ``fused_add(a, b)`` helper that compiles
``a + b`` as a single ``force=True`` graph (using a separate kernel-cache slot
from the plain norm).  CLIP's ``CLIPEncoderLayer.forward`` contains residual
adds (``residual + attn_out`` and ``residual + mlp_out``) that crash in eager
mode because the two operands come from separate OPWrapper graphs whose Spyre
EAs are incompatible.  ``fused_add`` fuses those adds into one Inductor graph
so the layout-propagation pass can assign compatible EAs to both operands at
once.

This is a drop-in subclass, not a global monkeypatch of
``torch.nn.LayerNorm`` -- only the specific boundary LayerNorms that actually
hit the crash (currently: CLIP's, patched in ``spyre_inference.multimodal.clip``)
should be swapped to it. Most ``LayerNorm`` call sites live inside a per-block
``torch.compile`` region and never take the crashing eager path in the first
place, so patching them too would be unnecessary blast radius onto unrelated
models.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
from vllm.platforms import current_platform

from .lazy_compile import CompileOutermost, maybe_compile


def _layer_norm_kernel(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    # Casting back to fp16 before the affine step -- rather than after -- hits a
    # torch-spyre layout limitation ("Multi-arg pointwise with mixed EA") when
    # multiplying by `weight`: the fp32 reduction's device layout doesn't
    # broadcast against a plain fp16 parameter. Keeping weight/bias in fp32 too
    # and casting only the final result avoids it.
    input_dtype = x.dtype
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        x_norm = x_norm * weight.float()
    if bias is not None:
        x_norm = x_norm + bias.float()
    return x_norm.to(input_dtype)


class SpyreLayerNorm(CompileOutermost, torch.nn.LayerNorm):
    """``torch.nn.LayerNorm`` that never invokes ``aten.layer_norm.default`` on Spyre.

    Has two compiled kernels:

    - ``_spyre_forward``: plain norm, cached in the shared ``spyre_compiled_kernel``
      slot from ``CompileOutermost``.
    - ``_spyre_fused_add``: a fused ``a + b`` add compiled with ``force=True``,
      cached in the separate ``_spyre_fused_kernel`` slot so it does not collide
      with the above.  Used by ``fused_add`` to prevent the "Multi-arg pointwise
      with mixed EA" crash on residual adds in CLIP encoder layers.
    """

    # Separate cache for the fused add kernel so it doesn't collide with
    # the shared ``spyre_compiled_kernel`` slot used by ``_spyre_forward``.
    _spyre_fused_kernel: Callable | None

    def arm_outer_compile(self) -> None:
        super().arm_outer_compile()
        if "_spyre_fused_kernel" not in self.__dict__:
            self._spyre_fused_kernel = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if input.device.type != "spyre":
            return super().forward(input)
        weight = self.weight if self.elementwise_affine else None
        bias = self.bias if self.elementwise_affine else None
        return self._spyre_forward(input, weight, bias)

    def fused_add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute ``a + b`` as a single ``force=True`` compiled graph on Spyre.

        CLIP encoder layers contain residual adds (``residual + attn_out`` and
        ``residual + mlp_out``) where the two operands originate from different
        OPWrapper graphs whose Spyre tensor EAs are incompatible.  Running the add
        in isolation through the OPWrapper triggers:

            Multi-arg pointwise with mixed EA: STANDARD input arg0_1 must broadcast …

        Compiling the add with ``force=True`` puts both operands into the same
        Inductor graph so the layout-propagation pass can assign compatible EAs to
        them simultaneously.

        Uses its own ``_spyre_fused_kernel`` cache slot (separate from the
        ``spyre_compiled_kernel`` used by ``_spyre_forward``) so the two compiled
        graphs do not overwrite each other.
        """
        if a.device.type != "spyre":
            return a + b
        if torch.compiler.is_compiling():
            # Already inside a compiled graph -- plain add is fine.
            return a + b
        if self._spyre_fused_kernel is None:
            from spyre_inference.v1.worker import compile_guard

            compile_guard.watch(
                SpyreLayerNorm._spyre_fused_add,
                "SpyreLayerNorm._spyre_fused_add",
            )
            self._spyre_fused_kernel = torch.compile(
                functools.partial(SpyreLayerNorm._spyre_fused_add, self),
                backend=current_platform.simple_compile_backend,
                fullgraph=True,
                dynamic=False,
            )
        return self._spyre_fused_kernel(a, b)

    @maybe_compile
    def _spyre_forward(
        self, x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return _layer_norm_kernel(x, weight, bias, self.eps)

    def _spyre_fused_add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b
