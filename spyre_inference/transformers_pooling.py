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


def _next_power_of_two(n: int) -> int:
    """Smallest power of two >= n (minimum 1)."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_encoder_inputs(
    batched_ids: torch.Tensor,   # [B_bucketed, L]  int64  CPU — already block-aligned
    seq_lens: list[int],         # real token count per row; len == B_real <= B_bucketed
    model_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build position_ids, tt_ids, and the SDPA additive mask on CPU.

    Replaces the preparation work inside ``prefill_encoder`` without
    redundant copies or Python-per-row loops where avoidable.

    Returns:
        position_ids  [B_bucketed, L]  int64  CPU
        tt_ids        [B_bucketed, L]  int64  CPU  (all-zero; moved to Spyre by wrapper)
        sdpa_mask     [B_bucketed, 1, L, L]  model_dtype  CPU
    """
    B_bucketed, L = batched_ids.shape
    B_real = len(seq_lens)

    # --- position_ids --------------------------------------------------
    # col < seq_len[b] selects real positions; multiply by col gives 0..L-1
    # for real tokens and 0 for pads (col[0]*True = 0 is correct for pos 0).
    col = torch.arange(L, dtype=torch.int64)           # [L]
    sl  = torch.tensor(seq_lens, dtype=torch.int64)    # [B_real]
    real_mask = col.unsqueeze(0) < sl.unsqueeze(1)     # [B_real, L] bool
    position_ids = torch.zeros(B_bucketed, L, dtype=torch.int64)
    position_ids[:B_real] = col.unsqueeze(0) * real_mask

    # --- token_type_ids ------------------------------------------------
    tt_ids = torch.zeros(B_bucketed, L, dtype=torch.int64)

    # --- SDPA additive mask --------------------------------------------
    # Shape [B_bucketed, 1, L, L]; bidirectional (is_causal=False).
    # Allowed pair (b, q, k): b < B_real and k < seq_lens[b].
    # Vectorised: broadcast seq_lens across the key axis.
    sdpa_mask = torch.full((B_bucketed, 1, L, L), -torch.inf, dtype=model_dtype)
    if B_real > 0:
        # key positions [L] < seq_lens [B_real] → allowed_k [B_real, L]
        allowed_k = col.unsqueeze(0) < sl.unsqueeze(1)  # [B_real, L]  bool
        # Expand to [B_real, 1, 1, L] then broadcast over the query axis
        sdpa_mask[:B_real] = torch.where(
            allowed_k[:, None, None, :],
            torch.zeros(1, dtype=model_dtype),
            torch.full((1,), -torch.inf, dtype=model_dtype),
        )

    return position_ids, tt_ids, sdpa_mask


def _rebatch(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    num_real_tokens: int,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pack flat ``input_ids[0:num_real_tokens]`` into ``[B_bucketed, L_max]``.

    Uses only the first ``num_real_tokens`` entries of ``input_ids`` and
    ``positions``, ignoring any bucket-padding zeros that follow.

    ``B_real``     = number of sequences (position resets within the real region).
    ``B_bucketed`` = next power of two >= ``B_real``; extra rows are zero-padded.
                     Bucketing the batch dimension reduces recompilations when
                     batch size varies across requests.
    ``L_max``      = ``max(positions[:num_real_tokens]) + 1``, rounded up to the
                     next ``_BLOCK_SIZE`` multiple.

    Tokens are placed at ``batched[seq_idx, pos]`` directly.

    Returns ``(batched_ids, attention_mask, B_real)`` all on CPU.
    ``B_real`` is needed by the caller to strip the batch-padding rows from the
    model output before flattening back to the packed token layout.
    """
    pos = positions[:num_real_tokens].tolist()
    ids = input_ids[:num_real_tokens]

    batch_size = 1
    for i in range(1, num_real_tokens):
        if pos[i] == 0:  # pos[i-1] > 0 guaranteed within the real region
            batch_size += 1

    batch_bucketed = _next_power_of_two(batch_size)
    raw_max = int(max(pos)) + 1  # 0-indexed; +1 gives token count
    max_len = ((raw_max + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE

    batched = torch.full((batch_bucketed, max_len), pad_id, dtype=input_ids.dtype)
    mask = torch.zeros((batch_bucketed, max_len), dtype=torch.long)

    seq_idx = 0
    for i in range(num_real_tokens):
        p = pos[i]
        if i > 0 and p == 0:
            seq_idx += 1
        batched[seq_idx, p] = ids[i]
        mask[seq_idx, p] = 1

    logger.debug(
        "rebatch: %d real tokens → [%d, %d] (B_real=%d)",
        num_real_tokens, batch_bucketed, max_len, batch_size,
    )
    return batched, mask, batch_size

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
        from hf_adapters.hf_common import get_model_dtype, prefill_encoder

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
        # Cache dtype and device once; used by _build_encoder_inputs / fast forward.
        self._model_dtype: torch.dtype = get_model_dtype(self.model)
        self._spyre_device: torch.device = next(self.model.parameters()).device

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
        # Fast path: runner pre-shaped the inputs in _prepare_inputs.
        batched_ids = kwargs.get("encoder_batched_ids")
        seq_lens: list[int] | None = kwargs.get("encoder_seq_lens")
        batch_size: int = kwargs.get("encoder_batch_size", 0)
        int_mask: torch.Tensor | None = None

        if batched_ids is None:
            # Fallback for dummy/warmup runs and direct forward() calls without runner.
            num_real_tokens: int = kwargs.get("num_scheduled_tokens", 0) or int(positions.shape[0])
            batched_ids, int_mask, batch_size = _rebatch(
                input_ids.cpu(), positions.cpu(), num_real_tokens, pad_id=self._pad_token_id
            )
            seq_lens = None

        if seq_lens is not None:
            # Full fast path: build all auxiliary tensors and call the backbone directly.
            position_ids, tt_ids, sdpa_mask = _build_encoder_inputs(
                batched_ids, seq_lens, self._model_dtype
            )
            dev = self._spyre_device
            last_hidden = self._run_backbone_forward(
                self.model,
                batched_ids.to(dev),
                sdpa_mask.to(dev),
                position_ids.to(dev),
                tt_ids.to(dev),
            )
        else:
            # Fallback: delegate to prefill_encoder (handles the integer mask from _rebatch).
            last_hidden = self._prefill_encoder(
                self._run_backbone_forward,
                self.model,
                batched_ids,
                int_mask,
            )
        # Drop batch-padding rows (rows batch_size..B_bucketed) before flattening
        # so the packed layout contains only real-sequence tokens.
        H = last_hidden.shape[-1]
        flat = last_hidden[:batch_size].reshape(-1, H)  # [batch_size * L_max, H]
        # Pad to num_tokens_padded so upstream's logit_indices never go out of
        # bounds (dummy runs produce fewer real rows than the padded token budget).
        num_tokens_padded = int(positions.shape[0])
        if flat.shape[0] < num_tokens_padded:
            flat = torch.cat([
                flat,
                torch.zeros(num_tokens_padded - flat.shape[0], H, dtype=flat.dtype, device=flat.device),
            ], dim=0)
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
