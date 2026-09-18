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

"""Custom vLLM model loader that delegates model preparation to hf_adapters.

Registered as load format ``"spyre_st"``.  Use it together with
``--runner pooling --model-impl transformers``::

    vllm serve ibm-granite/granite-embedding-278m-multilingual \\
        --runner pooling \\
        --model-impl transformers \\
        --load-format spyre_st \\
        --tensor-parallel-size 1 \\
        --port 8000 \\
        --max-model-len 512 \\
        --no-enable-prefix-caching \\
        --max-num-seqs 128

Instead of vLLM's standard ``initialize_model`` → ``load_weights`` pipeline
(``AutoModel.from_config`` on meta device, then streamed weight shards),
this loader calls ``AutoSpyreModel.from_pretrained`` which runs the full
hf-adapters preparation: custom weight loading, ``prepare_for_spyre``
(RoPE precomputation, compiled blocks, etc.), and ``move_model_to_spyre``.

The prepared HF model is wrapped in ``SpyreSentenceTransformerModel``, a thin
``nn.Module`` that satisfies ``VllmModelForPooling``:

* ``forward(input_ids, positions, ...)`` reshapes the flat vLLM token tensors
  into batched ``[B, L]`` form, reconstructs the ``attention_mask``, and calls
  ``prefill_embed`` or ``prefill_encoder``.
* ``pooler`` is a ``DispatchPooler`` identical to the one ``EmbeddingMixin``
  would build — all downstream Spyre pooling machinery uses it unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.interfaces_base import VllmModelForPooling

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.load import LoadConfig

logger = init_logger(__name__)


class SpyreSentenceTransformerModel(nn.Module, VllmModelForPooling):
    """Thin vLLM-compatible wrapper around an hf-adapters-prepared backbone.

    ``forward`` accepts the flat ``[T]`` tensors vLLM produces and reshapes
    them into the ``[B, L]`` form that ``prefill_embed`` / ``prefill_encoder``
    expect, reconstructing ``attention_mask`` from the per-request lengths
    encoded in ``positions`` (positions restart at 0 for each request).
    """

    is_pooling_model = True
    default_seq_pooling_type = "LAST"

    def __init__(self, hf_model: nn.Module, pooler: DispatchPooler) -> None:
        super().__init__()
        self.model = hf_model
        self.pooler = pooler

        # Route to the right prefill driver at construction time.
        # Encoder-only adapters (BERT, RoBERTa, …) set _is_encoder_only = True.
        from hf_adapters.auto_spyre_model import resolve_adapter_module

        adapter = resolve_adapter_module(hf_model.config.name_or_path or "")
        self._is_encoder_only = getattr(adapter, "_is_encoder_only", False)
        self._run_backbone_forward = adapter._run_backbone_forward

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        """Reshape flat vLLM inputs → [B, L] and call the prefill driver.

        vLLM concatenates all request tokens into a flat ``[T]`` tensor.
        For pooling the scheduler always completes full prompts, so we can
        recover per-request boundaries from ``positions``: a new request starts
        wherever ``positions[i] == 0`` (after the very first token).

        Returns:
            ``hidden_states`` on Spyre, shape ``[T, H]`` (flat, matching the
            vLLM convention that downstream poolers receive).
        """
        from hf_adapters.hf_common import prefill_embed, prefill_encoder

        # --- Reconstruct per-request boundaries from positions ---
        # positions = [0,1,...,L1-1, 0,1,...,L2-1, ...] — each request
        # restarts at 0.  Find the start index of each request.
        boundary = torch.cat(
            [torch.tensor([0]), (positions[1:] == 0).nonzero(as_tuple=False).squeeze(1) + 1]
        )
        req_lens = torch.diff(
            torch.cat([boundary, torch.tensor([input_ids.shape[0]])])
        ).tolist()

        # Pad all requests to the same length for batched [B, L] tensors.
        max_len = max(req_lens)
        batch_size = len(req_lens)
        ids_2d = torch.zeros(batch_size, max_len, dtype=torch.long)
        mask_2d = torch.zeros(batch_size, max_len, dtype=torch.long)
        offset = 0
        for b, length in enumerate(req_lens):
            ids_2d[b, :length] = input_ids[offset : offset + length].to(torch.long)
            mask_2d[b, :length] = 1
            offset += length

        # Call the appropriate prefill driver (CPU inputs, returns Spyre tensor).
        if self._is_encoder_only:
            h = prefill_encoder(
                self._run_backbone_forward,
                self.model,
                ids_2d,
                mask_2d,
            )
        else:
            h = prefill_embed(
                self._run_backbone_forward,
                self.model,
                ids_2d,
                mask_2d,
            )

        # h: [B, L, H] on Spyre → flatten back to [T, H] (vLLM's convention).
        # Crop each request to its real token count before stacking.
        rows = [h[b, :req_lens[b], :] for b in range(batch_size)]
        return torch.cat(rows, dim=0)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Weight loading was already done by AutoSpyreModel.from_pretrained;
        # this method exists only to satisfy the VllmModel protocol.
        return set()


class SpyreSentenceTransformerLoader(BaseModelLoader):
    """Model loader that uses ``hf_adapters.AutoSpyreModel`` for preparation.

    Bypasses vLLM's standard initialize-model → load-weights pipeline entirely.
    The hf-adapters preparation (``prepare_for_spyre``, ``move_model_to_spyre``)
    already places the model on Spyre and compiles the blocks, so
    ``process_weights_after_loading`` is still called for quant post-processing.
    """

    def download_model(self, model_config: "ModelConfig") -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(model_config.model, revision=model_config.revision)

    def load_weights(self, model: nn.Module, model_config: "ModelConfig") -> None:
        # Nothing to do: weights were loaded by AutoSpyreModel.from_pretrained.
        pass

    def load_model(
        self,
        vllm_config: "VllmConfig",
        model_config: "ModelConfig",
        prefix: str = "",
    ) -> nn.Module:
        from hf_adapters.auto_spyre_model import AutoSpyreModel

        model_config = vllm_config.model_config

        logger.info(
            "SpyreSentenceTransformerLoader: loading %s via hf_adapters",
            model_config.model,
        )

        # Full hf-adapters pipeline: load weights on CPU, prepare_for_spyre,
        # move_model_to_spyre.  The returned model is already on-device.
        hf_model = AutoSpyreModel.from_pretrained(
            model_config.model,
            dtype=model_config.dtype,
            trust_remote_code=model_config.trust_remote_code,
        )

        # Build the same DispatchPooler that EmbeddingMixin would create.
        pooler_config = model_config.pooler_config
        assert pooler_config is not None, (
            "pooler_config must be set; use --runner pooling"
        )
        pooler = DispatchPooler.for_embedding(pooler_config)

        model = SpyreSentenceTransformerModel(hf_model, pooler)

        # Run standard post-load weight processing (quant finalize, etc.).
        process_weights_after_loading(
            model, model_config, torch.device(vllm_config.device_config.device)
        )

        return model.eval()
