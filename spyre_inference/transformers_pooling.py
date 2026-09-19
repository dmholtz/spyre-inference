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

"""hf_adapters-backed Transformers embedding model for Spyre.

vLLM's Transformers backend (``--model-impl transformers``) routes encoder
pooling models to ``TransformersEmbeddingModel`` (= ``EmbeddingMixin +
LegacyMixin + Base``).  ``Base.__init__`` fuses Q/K/V into ``QKVParallelLinear``
and replaces attention with vLLM shims **before** weights load, which breaks
the hf_adapters ``prepare_for_spyre`` path that closes over the raw
``attention.self.{query,key,value}`` projections.

This module provides ``SpyreTransformersEmbeddingModel``, which bypasses
``Base`` entirely and instead:

1. Loads and prepares the model via ``AutoSpyreModel.from_pretrained``
   (hf_adapters path: load on CPU → ``prepare_for_spyre`` compiles blocks →
   ``move_model_to_spyre``).
2. Reconstructs per-sequence batched inputs from vLLM's flat packed tokens
   using position resets as sequence boundaries.
3. Calls ``prefill_encoder`` + the model-specific ``_run_backbone_forward``
   (resolved via ``resolve_adapter_module``).
4. Exposes a vLLM-compatible ``pooler`` so the runner's ``_pool`` path works
   unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


_BLOCK_SIZE = 64  # Spyre stick size (128 bytes / 2 bytes per fp16 element)


def _rebatch(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    num_real_tokens: int,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack flat ``input_ids[0:num_real_tokens]`` into ``[B, L_max]``.

    Uses only the first ``num_real_tokens`` entries of ``input_ids`` and
    ``positions``, ignoring any bucket-padding zeros that follow.

    ``B``     = number of sequences (position resets to 0 within the real
                region, each preceded by a non-zero position).
    ``L_max`` = ``max(positions[:num_real_tokens]) + 1``, rounded up to the
                next ``_BLOCK_SIZE`` multiple.

    Tokens are placed at ``batched[seq_idx, pos]`` directly — no offset
    accumulation needed.

    Returns ``(batched_ids, attention_mask)`` both on CPU, mirroring the
    layout that ``prefill_encoder`` expects.
    """
    pos = positions[:num_real_tokens].tolist()
    ids = input_ids[:num_real_tokens]

    batch_size = 1
    for i in range(1, num_real_tokens):
        if pos[i] == 0:  # pos[i-1] > 0 guaranteed within the real region
            batch_size += 1

    raw_max = int(max(pos)) + 1  # 0-indexed; +1 gives token count
    max_len = ((raw_max + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE

    batched = torch.full((batch_size, max_len), pad_id, dtype=input_ids.dtype)
    mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    seq_idx = 0
    for i in range(num_real_tokens):
        p = pos[i]
        if i > 0 and p == 0:
            seq_idx += 1
        batched[seq_idx, p] = ids[i]
        mask[seq_idx, p] = 1

    print("rebatch: input [%d] → batched [%d, %d]", num_real_tokens, batch_size, max_len)
    return batched, mask

from vllm.model_executor.models.transformers import TransformersEmbeddingModel


class SpyreTransformersEmbeddingModel(TransformersEmbeddingModel):
    """Encoder pooling model for ``--model-impl transformers`` on Spyre.

    Loaded via hf_adapters so the backbone uses compiled blocks and the
    Spyre-native bidirectional attention path, not vLLM's encoder attention shim.
    """

    is_pooling_model = True
    default_seq_pooling_type = "CLS"
    attn_type = "encoder_only"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)

        from hf_adapters.auto_spyre_model import AutoSpyreModel, resolve_adapter_module
        from hf_adapters.hf_common import prefill_encoder

        model_path = vllm_config.model_config.model
        self._adapter_module = resolve_adapter_module(model_path)
        self._run_backbone_forward = self._adapter_module._run_backbone_forward
        self._prefill_encoder = prefill_encoder

        logger.info(
            "SpyreTransformersEmbeddingModel: loading %s via hf_adapters (%s)",
            model_path,
            self._adapter_module.__name__,
        )
        self.model = AutoSpyreModel.from_pretrained(model_path)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = DispatchPooler.for_embedding(pooler_config)

        self._pad_token_id = int(
            getattr(vllm_config.model_config.hf_config, "pad_token_id", 0) or 0
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        num_real_tokens: int = kwargs.get("num_scheduled_tokens", 0) or int(positions.shape[0])
        batched_ids, attention_mask = _rebatch(
            input_ids.cpu(), positions.cpu(), num_real_tokens, pad_id=self._pad_token_id
        )
        # prefill_encoder returns [B, L, H] on Spyre (with block-pad cropped).
        last_hidden = self._prefill_encoder(
            self._run_backbone_forward,
            self.model,
            batched_ids,
            attention_mask,
        )
        # Flatten [B, L, H] → [B*L, H] and crop to the real token count.
        # During dummy/warmup runs the positions buffer is all zeros so _rebatch
        # produces fewer rows (B=1, L=64) than num_tokens_padded.  Pad with zeros
        # so that upstream's logit_indices can always index into a full-sized
        # [num_tokens_padded, H] tensor without going out of bounds.
        flat = last_hidden.reshape(-1, last_hidden.shape[-1])
        num_tokens_padded = int(positions.shape[0])
        if flat.shape[0] < num_tokens_padded:
            pad = torch.zeros(
                num_tokens_padded - flat.shape[0],
                flat.shape[1],
                dtype=flat.dtype,
                device=flat.device,
            )
            flat = torch.cat([flat, pad], dim=0)
        return flat[:num_tokens_padded]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        # Weights are already loaded by AutoSpyreModel.from_pretrained in __init__.
        # Return None so the loader skips its weight-coverage check (line 437 in
        # default_loader.py: ``loaded_weights is not None`` gates track_weights_loading).
        pass

    def make_empty_intermediate_tensors(self, *args, **kwargs) -> IntermediateTensors:
        raise NotImplementedError(
            "SpyreTransformersEmbeddingModel does not support pipeline parallelism"
        )

    def create_attention_instances(self) -> dict[int, Attention]:
        # Return an empty dict if the underlying model handles attention natively
        # without registering vLLM Attention layers for KV cache management.
        return {}

    def recursive_replace(self) -> None:
        # No-op: Do not replace linears, norms, convolutions, or fusions.
        # All weights remain standard PyTorch modules on the target device
        # once init_parameters() finishes.
        pass

    def _patch_config():
        print("patch nothing")


# Using_transformers_backend() compares _ModelInfo.architecture, which is
# model_cls.__name__, against "TransformersEmbeddingModel", so the subclass
# has to keep answering to that name.
SpyreTransformersEmbeddingModel.__name__ = "TransformersEmbeddingModel"
