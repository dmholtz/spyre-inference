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
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _seq_lengths_from_positions(positions: torch.Tensor) -> list[int]:
    """Recover per-sequence lengths from flat packed position ids.

    vLLM sets positions as ``[0, 1, ..., L1-1, 0, 1, ..., L2-1, ...]``.
    A new sequence starts wherever ``positions[i] <= positions[i-1]``
    (reset to 0) or at token 0 of the whole batch.
    """
    pos = positions.tolist()
    lengths: list[int] = []
    run = 1
    for i in range(1, len(pos)):
        if pos[i] <= pos[i - 1]:
            lengths.append(run)
            run = 1
        else:
            run += 1
    lengths.append(run)
    return lengths


def _rebatch(
    input_ids: torch.Tensor,
    seq_lengths: list[int],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack flat ``input_ids [T]`` into ``[B, L_max]`` with right-padding.

    Returns ``(batched_ids, attention_mask)`` both on CPU, mirroring the
    layout that ``prefill_encoder`` expects.
    """
    batch_size = len(seq_lengths)
    max_len = max(seq_lengths)
    batched = torch.full((batch_size, max_len), pad_id, dtype=input_ids.dtype)
    mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    offset = 0
    for i, length in enumerate(seq_lengths):
        batched[i, :length] = input_ids[offset : offset + length]
        mask[i, :length] = 1
        offset += length
    return batched, mask


class SpyreTransformersEmbeddingModel(nn.Module, VllmModelForPooling):
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
        seq_lengths = _seq_lengths_from_positions(positions)
        batched_ids, attention_mask = _rebatch(
            input_ids.cpu(), seq_lengths, pad_id=self._pad_token_id
        )
        # prefill_encoder returns [B, L, H] on Spyre (with block-pad cropped).
        last_hidden = self._prefill_encoder(
            self._run_backbone_forward,
            self.model,
            batched_ids,
            attention_mask,
        )
        # Flatten back to [T, H] so the downstream pooler's cursor arithmetic
        # (CLS/LAST row indices, MEAN token ranges) works on the packed layout.
        return last_hidden.reshape(-1, last_hidden.shape[-1])[: int(positions.shape[0])]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Weights are already loaded by AutoSpyreModel.from_pretrained in __init__.
        return set()

    def make_empty_intermediate_tensors(self, *args, **kwargs) -> IntermediateTensors:
        raise NotImplementedError(
            "SpyreTransformersEmbeddingModel does not support pipeline parallelism"
        )


# Using_transformers_backend() compares _ModelInfo.architecture, which is
# model_cls.__name__, against "TransformersEmbeddingModel", so the subclass
# has to keep answering to that name.
SpyreTransformersEmbeddingModel.__name__ = "TransformersEmbeddingModel"
