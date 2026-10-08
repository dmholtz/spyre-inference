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

"""Spyre-specific Conv2d implementation (Pixtral/Ministral vision patch embed).

vLLM lowers a patch conv to im2col + GEMM, whose on-device reshape produces a
sub-stick `copy_from_d2d` expression torch-spyre cannot lay out for patch grids
coprime with the 64-wide stick. So run the real `F.conv2d` on-card instead, with
the weight and input placed into explicit `SpyreTensorLayout`s. Layout tuples are
derived from shapes, so any out-channel count and image size work.

For shapes where those tiled layouts do not apply (batch > 1, C_in > 64, or
out_channels not a multiple of 64) a fallback path is used.  The fallback was
previously ``F.conv2d`` on device, which decomposes via ``conv2d_via_bmm_decomp``
into three independent host round trips (spyre::unfold for the activation plus
spyre::reshape_via_cpu for weight and bias).

Strategy C eliminates the weight and bias round trips by pre-computing and
placing them on device once in ``process_weights_after_loading``.  At inference
the fallback path calls ``spyre::unfold`` (the single inescapable activation
round trip) and then a compiled on-device matmul + bias-add using the
pre-placed ``_w_2d_dev`` / ``_bias_3d_dev`` tensors, reducing three host round
trips to one.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.platforms import current_platform

from .lazy_compile import CompileOutermost, maybe_compile
from .utils import convert

logger = init_logger(__name__)


def _layouts_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the layout tuples below apply: one image, in-channels within a stick,
    out-channels a whole number of sticks.

    True for a Pixtral patch embed, not for convs in general — and this class is
    registered OOT for *every* `Conv2dLayer`, so fall back rather than assert.
    """
    if x.dim() != 4 or weight.dim() != 4:
        return False
    b, c = x.shape[0], x.shape[1]
    return b == 1 and c <= 64 and weight.shape[0] % 64 == 0


def _weight_layout(weight: torch.Tensor):
    """SpyreTensorLayout for a conv weight (O, C, K1, K2), sticked on out-channels.

    The stick walks the out-channel dim (host stride C*K1*K2), tiling it into O//64.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    o, c, k1, k2 = weight.shape
    assert o % 64 == 0, f"conv out_channels {o} must be a multiple of the 64-wide stick"
    return SpyreTensorLayout(
        [k2, k1, o // 64, c, 64],
        [1, k2, c * k1 * k2 * 64, k1 * k2, c * k1 * k2],
        get_device_dtype(weight.dtype),
    )


def _input_layout(x: torch.Tensor):
    """SpyreTensorLayout for a conv input (1, C, H, W), sticked on in-channels.

    The stick walks the channel dim (host stride H*W), padding C up to a full stick.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    b, c, h, w = x.shape
    assert b == 1, f"conv input batch {b} != 1 (Pixtral feeds one image at a time)"
    assert c <= 64, f"conv in_channels {c} must fit in one 64-wide stick"
    return SpyreTensorLayout(
        [w, h, 1, 1, 64],
        [1, w, -1, c * h * w, h * w],
        get_device_dtype(x.dtype),
    )


@Conv2dLayer.register_oot(name="Conv2dLayer")
class SpyreConv2d(CompileOutermost, Conv2dLayer):
    """Out-of-tree Conv2d for Spyre: `F.conv2d` on-card with explicit tiled layouts.

    Spyre needs static shapes, so the kernel recompiles per distinct (H, W). The
    platform lifts dynamo's recompile limits, so there is no eager fallback, but every
    new resolution costs a full compile mid-request — bucket or resize images if a
    workload uses many resolutions.
    """

    # Per-(H, W) recompiles are this layer's contract, so the compile guard must not
    # report them; warmup cannot enumerate every image resolution.
    allow_inference_recompiles = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._w_dev: torch.Tensor | None = None
        # Strategy C: pre-placed fallback weight/bias (populated by
        # process_weights_after_loading for shapes outside _layouts_supported).
        self._w_2d_dev: torch.Tensor | None = None
        self._bias_3d_dev: torch.Tensor | None = None
        self._compiled_conv_via_matmul: Callable | None = None

    @maybe_compile
    def _conv_native(self, x: torch.Tensor, w: torch.Tensor, bias) -> torch.Tensor:
        return F.conv2d(
            x,
            w,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def _conv_via_matmul(
        self,
        patches: torch.Tensor,
        w_2d: torch.Tensor,
        bias_3d: torch.Tensor | None,
        h_out: int,
        w_out: int,
    ) -> torch.Tensor:
        """On-device matmul + bias-add for the pre-loaded fallback path.

        ``patches``  : (N, C_in*K1*K2, H_out*W_out)  — output of spyre::unfold
        ``w_2d``     : (C_out, C_in*K1*K2)            — pre-reshaped at load time
        ``bias_3d``  : (1, C_out, 1) or None          — pre-reshaped at load time
        ``h_out``, ``w_out``: output spatial dims (static per compile graph)

        Returns (N, C_out, H_out, W_out).
        """
        N = patches.shape[0]
        # Broadcast weight over batch: (1, C_out, C_in*K) -> (N, C_out, C_in*K).
        # clone() materialises the expand so the matmul sees a contiguous layout.
        out = torch.matmul(w_2d.unsqueeze(0).expand(N, -1, -1).clone(), patches)
        # out: (N, C_out, H_out*W_out)
        if bias_3d is not None:
            out = out + bias_3d
        return out.reshape(N, self.out_channels, h_out, w_out)

    def process_weights_after_loading(self) -> None:
        """Place conv weights into their tiled / pre-reshaped layouts once after load.

        Happy path (``_layouts_supported``): places the weight into the tiled
        ``_weight_layout`` as before.

        Fallback path (shapes outside ``_layouts_supported``): additionally
        pre-reshapes weight to ``(C_out, C_in*K1*K2)`` and bias to
        ``(1, C_out, 1)`` and places them on device.  At inference these are
        passed directly to the compiled matmul, eliminating the per-step
        ``spyre::reshape_via_cpu`` round trips that ``conv2d_via_bmm_decomp``
        would otherwise emit for weight and bias (Strategy C).
        """
        if self.weight.device.type != "spyre":
            return

        # Happy-path weight (tiled layout for on-card F.conv2d).
        if self._w_dev is None:
            w_cpu = convert(self.weight.detach(), device="cpu")
            self._w_dev = convert(w_cpu, device="spyre", device_layout=_weight_layout(w_cpu))

        # Fallback-path weight + bias (flat 2D / 3D, default layout).
        # groups == 1 is the only case conv2d_via_bmm_decomp's non-depthwise branch
        # handles; depthwise (C_in == groups == C_out) routes to spyre::conv2d_with_bias
        # and never reaches this code path.
        if self._w_2d_dev is None and self.groups == 1:
            C_out, C_in, K1, K2 = self.weight.shape
            w_cpu = convert(self.weight.detach(), device="cpu")
            self._w_2d_dev = convert(
                w_cpu.reshape(C_out, C_in * K1 * K2), device="spyre"
            )
            if self.bias is not None and self._bias_3d_dev is None:
                b_cpu = convert(self.bias.detach(), device="cpu")
                self._bias_3d_dev = convert(
                    b_cpu.reshape(1, C_out, 1), device="spyre"
                )

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        # `_forward_conv`, not `forward_native`: a patch embed sets `enable_linear`, so
        # `forward_native` picks the unfold/reshape path this class exists to avoid.
        if x.device.type != "spyre":
            # The tiled layouts move a tensor onto the card; applying them to a
            # CPU input is the opposite of what the caller asked for.
            return self._forward_conv(x)
        if not _layouts_supported(x, self.weight):
            logger.warning_once(
                "Spyre conv2d: shape %s (weight %s) outside the tiled-layout "
                "assumptions (batch 1, in_channels <= 64, out_channels %% 64 == 0); "
                "falling back to im2col+matmul with pre-loaded weight.",
                tuple(x.shape),
                tuple(self.weight.shape),
            )
            if self._w_2d_dev is not None:
                # Strategy C: weight and bias were placed on device at load time.
                # Only the activation needs a host round trip (spyre::unfold).
                _, _, H_in, W_in = x.shape
                K1, K2 = self.kernel_size
                S1, S2 = self.stride
                P1, P2 = self.padding
                D1, D2 = self.dilation
                H_out = (H_in + 2 * P1 - D1 * (K1 - 1) - 1) // S1 + 1
                W_out = (W_in + 2 * P2 - D2 * (K2 - 1) - 1) // S2 + 1
                patches = torch.ops.spyre.unfold(
                    x,
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    padding=self.padding,
                    stride=self.stride,
                )
                if not torch.compiler.is_compiling() and self.spyre_compile_enabled:
                    if self._compiled_conv_via_matmul is None:
                        self._compiled_conv_via_matmul = torch.compile(
                            self._conv_via_matmul,
                            backend=current_platform.simple_compile_backend,
                            fullgraph=True,
                            dynamic=False,
                        )
                    return self._compiled_conv_via_matmul(
                        patches, self._w_2d_dev, self._bias_3d_dev, H_out, W_out
                    )
                return self._conv_via_matmul(
                    patches, self._w_2d_dev, self._bias_3d_dev, H_out, W_out
                )
            # _w_2d_dev is None only before process_weights_after_loading has run
            # (e.g. a CPU-device forward during model construction) or for grouped
            # convs where the fallback pre-load is not implemented.  Route through
            # aten.convolution so conv2d_via_bmm_decomp handles it.
            return self._forward_conv(x)
        logger.info_once("Spyre conv2d: on-card F.conv2d with tiled layouts")
        # Via CPU: CPU->spyre is the tested entry path, and a device-side
        # restickify would hit the same unsupported layout.
        x_cpu = convert(x, device="cpu")
        x_dev = convert(x_cpu, device="spyre", device_layout=_input_layout(x_cpu))
        assert self._w_dev is not None, "Conv weights must be prepared after model loading."
        return self._conv_native(x_dev, self._w_dev, self.bias)
