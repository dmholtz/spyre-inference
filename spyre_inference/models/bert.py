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

"""Spyre adaptations for vLLM BERT-family pooling models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.bert import (
    BertEmbedding,
    BertEmbeddingModel,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
    BertSelfAttention,
    BertSpladeSparseEmbeddingModel,
)

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._token_type import (
    SpyreTokenTypeEmbedding,
    SpyreTokenTypeModel,
)
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    SpyreEncoderAttentionImpl,
    _align_up,
    _b1_dense_attention,
    _cached_encoder_workspace,
    _ensure_encoder_pack,
    _packed_masked_attention,
    gather_unpack,
    scatter_pack_hidden,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class SpyreBertEmbedding(SpyreTokenTypeEmbedding, BertEmbedding):
    """``BertEmbedding`` reading segment ids from the side buffer."""

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = (
            inputs_embeds
            + self.spyre_token_type_embeddings(input_ids)
            + self.position_embeddings(position_ids)
        )
        return self.LayerNorm(embeddings)


class SpyreBertSelfAttention(BertSelfAttention):
    """``BertSelfAttention`` that packs ``hidden_states`` before the QKV projection.

    Replaces three per-head scatter_pack calls (Q, K, V) with one scatter of the
    flat ``[T, hidden]`` input. After packing, the fused linear runs on the
    uniform ``[B*L, hidden]`` layout so its output is already dense — no strided
    views, no pre-copy. SDPA and unpack run once. Net saving: 2x scatter +
    2x permute.contiguous.

    The attention backend (``SpyreEncoderAttentionImpl``) is bypassed entirely:
    ``self.attn(q, k, v)`` is never called. Pack metadata is built via
    ``_ensure_encoder_pack`` (same call the backend makes; idempotent, so
    subsequent layers reuse it).
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        fwd_ctx = get_forward_context()
        attn_metadata_raw = fwd_ctx.attn_metadata
        if isinstance(attn_metadata_raw, list):
            attn_metadata = attn_metadata_raw[0][self.attn.layer_name]
        else:
            attn_metadata = attn_metadata_raw[self.attn.layer_name]

        if attn_metadata is None:
            # Warmup path — upstream backend handles dummy runs.
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            return self.attn(q, k, v)

        # Retrieve the impl for config fields (cached once at construction).
        attn_layer = fwd_ctx.no_compile_layers[self.attn.layer_name]
        impl: SpyreEncoderAttentionImpl = attn_layer.impl  # type: ignore[assignment]

        padded_tokens = hidden_states.shape[0]
        n = attn_metadata.num_actual_tokens
        target_device = hidden_states.device
        hidden_size = hidden_states.shape[-1]

        _ensure_encoder_pack(
            attn_metadata,
            padded_tokens=padded_tokens,
            n=n,
            query=hidden_states,  # only .dtype is read
            num_kv_heads=self.num_kv_heads,
            target_device=target_device,
            cached_encoder_shapes=impl._cached_encoder_shapes,
            cached_max_num_seqs=impl._cached_max_num_seqs,
            cached_max_model_len=impl._cached_max_model_len,
            cached_max_num_batched_tokens=impl._cached_max_num_batched_tokens,
        )

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        if attn_metadata.encoder_fused_sdpa:
            # B=1, T==L, no live padding: skip scatter entirely.
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # split() returns strided views; reshape() copies where view() would crash.
            q3 = q.reshape(padded_tokens, self.num_heads, self.head_dim)
            k3 = k.reshape(padded_tokens, self.num_kv_heads, self.head_dim)
            v3 = v.reshape(padded_tokens, self.num_kv_heads, self.head_dim)
            result3 = _b1_dense_attention(
                q3, k3, v3,
                impl.scale,
                self.num_kv_heads != self.num_heads,
                _align_up(self.head_dim),  # head_size_padded
                self.head_dim,             # true head_size
            )
            return result3.reshape(padded_tokens, hidden_size)

        batch = attn_metadata.encoder_pack_batch
        aligned_len = attn_metadata.encoder_pack_len
        assert batch is not None and aligned_len is not None

        # One scatter of hidden_states [T, hidden] → [B, L, hidden].
        rows = batch * aligned_len + 1
        hs_ws = _cached_encoder_workspace(
            attn_metadata,
            "encoder_hs_workspace",
            rows,
            1,           # num_heads=1 (hidden is unsqueezed inside scatter_pack_hidden)
            hidden_size,
            hidden_states.dtype,
            target_device,
        )
        hs_packed = scatter_pack_hidden(
            hidden_states,
            attn_metadata.encoder_q_pack_idx,  # same dest as Q (same token positions)
            batch,
            aligned_len,
            workspace=hs_ws,
        )  # [B, L, hidden_size]

        # QKV projection on the packed, uniform layout.
        hs_flat = hs_packed.reshape(batch * aligned_len, hidden_size)
        qkv_packed, _ = self.qkv_proj(hs_flat)  # [B*L, q+k+v]
        q_packed, k_packed, v_packed = qkv_packed.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        # split() views are strided; reshape() materialises dense [B, H, L, D] copies.
        q4 = q_packed.reshape(batch, aligned_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k4 = k_packed.reshape(batch, aligned_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v4 = v_packed.reshape(batch, aligned_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        # q4/k4/v4 are [B, H, L, D] — the shape _packed_masked_attention expects.

        key_pad_mask = attn_metadata.encoder_key_pad_mask
        assert key_pad_mask is not None
        attn_out = _packed_masked_attention(q4, k4, v4, key_pad_mask, impl.scale)
        # attn_out is [B, H, L, D].

        unpack_idx = attn_metadata.encoder_unpack_idx
        assert unpack_idx is not None
        result3 = gather_unpack(attn_out, unpack_idx, self.head_dim)
        # result3 is [T, H, D]; reshape to [T, hidden_size].
        result = result3.reshape(padded_tokens, hidden_size)
        if result.dtype != hidden_states.dtype:
            result = convert(result, dtype=hidden_states.dtype)
        return result


class SpyreBertSelfAttentionMixin:
    """Walk all ``BertSelfAttention`` modules and replace them in-place.

    Applied once after ``super().__init__``. Only activates on Spyre — the
    replacement is a no-op elsewhere because ``SpyreBertSelfAttention``
    reads from ``get_forward_context()`` which only has ``SpyreAttentionMetadata``
    on the Spyre platform. For non-Spyre paths the fallback in
    ``SpyreBertSelfAttention.forward`` delegates to the upstream ``self.attn``
    call, so correctness is preserved.
    """

    def _replace_bert_self_attention(self) -> None:
        for module in self.modules():  # type: ignore[attr-defined]
            if type(module) is BertSelfAttention:
                module.__class__ = SpyreBertSelfAttention

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)  # type: ignore[call-arg]
        self._replace_bert_self_attention()


class SpyreBertEmbeddingMixin:
    """Inject the Spyre embedding through ``BertEmbeddingModel._build_model``."""

    def _build_model(self, vllm_config: VllmConfig, prefix: str = "") -> BertModel:
        return BertModel(vllm_config=vllm_config, prefix=prefix, embedding_class=SpyreBertEmbedding)


class SpyreBertEmbeddingModel(SpyreBertSelfAttentionMixin, SpyreBertEmbeddingMixin, BertEmbeddingModel):
    pass


class SpyreBertSpladeSparseEmbeddingModel(
    SpyreBertSelfAttentionMixin, SpyreBertEmbeddingMixin, BertSpladeSparseEmbeddingModel
):
    pass


class SpyreBertForSequenceClassification(
    SpyreBertSelfAttentionMixin, SpyreTokenTypeModel, BertForSequenceClassification
):
    spyre_embedding_class = SpyreBertEmbedding


class SpyreBertForTokenClassification(
    SpyreBertSelfAttentionMixin, SpyreTokenTypeModel, BertForTokenClassification
):
    spyre_embedding_class = SpyreBertEmbedding


class SpyreBertForMaskedLM(SpyreBertSelfAttentionMixin, SpyreTokenTypeModel, BertForMaskedLM):
    spyre_embedding_class = SpyreBertEmbedding
