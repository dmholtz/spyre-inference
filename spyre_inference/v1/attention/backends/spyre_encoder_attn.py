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

"""Encoder-only (bidirectional) self-attention for Spyre without a KV cache.

Selected by ``TorchSpyrePlatform.get_attn_backend_cls`` for ENCODER/ENCODER_ONLY
layers. Operates on direct Q/K/V tensors rather than the paged KV-cache path.

Instead of full-batch scatter-packing into [B, H, L, D] workspaces and gather-unpacking,
the attention processes each sequence individually in a loop over B=1 sequences:
each sequence's valid token slice is padded to its own stick/bucket length and executed
with compiled B=1 masked/dense attention.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionLayer

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
    _call_kernel,
    is_warmup_complete,
)
from spyre_inference.v1.attention.ops.layout import slot_major_kv_layout
from spyre_inference.v1.pool import select_rows
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    default_encoder_len_buckets,
    next_bucket,
    pick_encoder_attention_shape,
    pooling_warmup_shapes,
)

logger = init_logger(__name__)

# Pad seq length *and* head dim to the Spyre stick (64 fp16 elements).
# L-aligned keeps P·V's K stick-aligned; D-aligned keeps QKᵀ's K stick-aligned
# so Inductor never enters insert_bmm_padding (torch-spyre KeyError: 'val' on
# FX nodes missing meta["val"] when padding MiniLM's head_size=32).
ENCODER_SEQ_ALIGNMENT = 64


def _align_up(n: int, align: int = ENCODER_SEQ_ALIGNMENT) -> int:
    return max(align, (n + align - 1) // align * align)


def host_pack_indices(
    q_starts: list[int],
    lengths: list[int],
    aligned_len: int,
    pad_row: int,
) -> torch.Tensor:
    """Build ``[B, L]`` int64 row indices; pad slots point at ``pad_row``."""
    batch = len(q_starts)
    indices = torch.full((batch, aligned_len), pad_row, dtype=torch.int64)
    for s, (start, length) in enumerate(zip(q_starts, lengths)):
        if length > 0:
            indices[s, :length] = torch.arange(start, start + length, dtype=torch.int64)
    return indices


def _content_query_lens(
    qsl_lens: list[int],
    kv_lens: list[int],
    num_actual_tokens: int | None = None,
) -> list[int]:
    """Per-seq counts for dest/mask. ``query_start_loc`` can include 1D body pad.

    When B=1 qsl and ``seq_lens`` are both the body bucket, ``num_actual_tokens``
    (unpadded scheduled count) still marks pad: ``min(qsl, seq, n)``.

    B>1 does not redistribute. The runner 1D-pads the concatenated body
    (suffix after the last sequence); per-seq padded qsl cannot occur. Shrinking
    the tail would turn real ``[5, 12]`` reported as ``[32, 32]`` into ``[17, 0]``.
    """
    out = [min(int(q), int(k)) for q, k in zip(qsl_lens, kv_lens)]
    if not out or num_actual_tokens is None:
        return out
    n = int(num_actual_tokens)
    if len(out) == 1:
        out[0] = min(out[0], n)
        return out
    excess = sum(out) - n
    if excess > 0:
        raise AssertionError(
            f"B>1 query_start_loc/seq_lens include {excess} pad tokens "
            f"(lens={out}, num_actual_tokens={n}). Per-seq padded qsl is "
            "unsupported; 1D body pad is a suffix after the last sequence."
        )
    return out


def dummy_pack_row(lengths: list[int], aligned_len: int) -> int:
    """First pad slot in the ``[B, L]`` grid, or ``0`` when the grid is full.

    Tests only. Serve writes body-pad to an extra workspace row (``B×L``),
    never ``0``: a full grid plus a 1D body bucket ``> B×L`` would otherwise
    overwrite CLS.
    """
    for s, length in enumerate(lengths):
        if length < aligned_len:
            return s * aligned_len + length
    return 0


def host_scatter_pack_dest(
    q_starts: list[int],
    lengths: list[int],
    aligned_len: int,
    num_src_rows: int,
    dummy_row: int,
) -> torch.Tensor:
    """``[num_src_rows]`` packed-row dest for each source token.

    Real tokens write ``s * L + pos``. Body-pad rows write ``dummy_row``
    (serve: extra workspace row ``B×L``, not a real token).
    """
    dest = torch.full((num_src_rows,), dummy_row, dtype=torch.int64)
    for s, (start, length) in enumerate(zip(q_starts, lengths)):
        if length > 0:
            dest[start : start + length] = torch.arange(
                s * aligned_len, s * aligned_len + length, dtype=torch.int64
            )
    return dest


def host_unpack_indices(
    q_starts: list[int],
    query_lens: list[int],
    aligned_len: int,
    num_tokens: int,
) -> torch.Tensor:
    """Build ``[T]`` int64 indices from flat padded ``[B*L]`` back to tokens.

    ``num_tokens`` may exceed the real count; unfilled entries stay ``0``
    (a safe row to read — nothing downstream reads those output rows).
    """
    indices = torch.zeros(num_tokens, dtype=torch.int64)
    for s, (start, length) in enumerate(zip(q_starts, query_lens)):
        if length > 0:
            base = s * aligned_len
            indices[start : start + length] = torch.arange(base, base + length, dtype=torch.int64)
    return indices


def _pad_head_dim_to_stick(flat: torch.Tensor, head_size_padded: int) -> torch.Tensor:
    """Pad last dim to a stick. MiniLM ``[T,H,32]`` cannot ``F.pad`` on Spyre."""
    head_size = flat.shape[-1]
    if head_size == head_size_padded:
        return flat
    device = flat.device
    if device.type == "spyre":
        flat = convert(flat, "cpu")
    flat = F.pad(flat, (0, head_size_padded - head_size))
    if device.type == "spyre":
        flat = convert(flat, device)
    return flat


def _is_identity_row_map(indices: torch.Tensor, num_rows: int) -> bool:
    """True when ``indices`` is ``0 .. num_rows-1`` (no pad gather). Host only."""
    if indices.numel() != num_rows or indices.device.type != "cpu":
        return False
    flat = indices.reshape(-1)
    return bool(torch.equal(flat, torch.arange(num_rows, dtype=flat.dtype)))


def _is_b1_dense_body(batch: int, num_src: int, aligned_len: int) -> bool:
    """Single sequence already shaped ``[L, H, D]`` (token bucket == SDPA L)."""
    return batch == 1 and num_src == aligned_len


def _is_b1_fused_sdpa(batch: int, padded_tokens: int, aligned_len: int, real_len: int) -> bool:
    """Compiled permute+SDPA is only legal when the mask is dense (no live pad)."""
    return _is_b1_dense_body(batch, padded_tokens, aligned_len) and real_len == aligned_len


def _index_copy_kernel(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Tiny mutation, compiled alone — do not fuse with SDPA."""
    dst.index_copy_(0, index, src)
    return dst


_CompiledFn = Callable[..., torch.Tensor]

_compiled_kernels: dict[Callable[..., torch.Tensor], _CompiledFn] = {}


def _compile_if_spyre(kernel: _CompiledFn, device_type: str) -> _CompiledFn:
    """Compile ``kernel`` once on Spyre. CPU always runs ``kernel``.

    Memo is keyed on the Python function, so packed QK, P·V, index_copy_, and
    the two B=1 SDPA kernels each compile once without caller write-back.
    """
    if device_type != "spyre":
        return kernel
    compiled = _compiled_kernels.get(kernel)
    if compiled is None:
        compiled = cast(_CompiledFn, torch.compile(kernel, dynamic=False))
        _compiled_kernels[kernel] = compiled
    return compiled


def _b1_sdpa_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """``[T,H,D]`` → SDPA ``[1,H,T,D]`` → ``[T,H,D]``. No mask, no eager ``contiguous``.

    Full-bucket fused path only (``real_len == L``). A zeros ``attn_mask`` is
    still an H2D + per-layer add; Spyre compiled SDPA also drops a live pad
    mask, so pad stays on the packed path.
    """
    q = query.unsqueeze(0).transpose(1, 2)
    k = key.unsqueeze(0).transpose(1, 2)
    v = value.unsqueeze(0).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)
    t, h, d = query.shape
    return out.transpose(1, 2).reshape(t, h, d)


def _b1_sdpa_kernel_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    q = query.unsqueeze(0).transpose(1, 2)
    k = key.unsqueeze(0).transpose(1, 2)
    v = value.unsqueeze(0).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale, enable_gqa=True)
    t, h, d = query.shape
    return out.transpose(1, 2).reshape(t, h, d)


def host_key_pad_mask(mask: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """CPU ``[B,1,L,L]`` → ``[B*KV, 1, 1, L]``, broadcast over the query axis.

    The same key-pad row applies to every query row, so only the KV axis needs
    materialising. This shape was previously rejected because an *eager* Spyre
    add cannot broadcast ``1 → L`` on the query axis (no stick-scatter), which
    forced a dense ``[B*KV, 1, L, L]``. The add now happens inside the compiled
    ``_packed_pv``, where Inductor broadcasts it, so the dense form is no longer
    needed and it was expensive.

    Broadcasting also covers the GQA head axis (``G``), so the caller no longer
    needs an eager ``expand_as(...).contiguous()`` either.
    """
    key = mask[:, :, :1, :]
    batch, _, _, length = key.shape
    return (
        key.expand(batch, num_kv_heads, 1, length)
        .reshape(batch * num_kv_heads, 1, 1, length)
        .contiguous()
    )


def _b1_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """B=1 masked attention kernel for a single sequence with key-pad mask.

    Shapes:
      query: [1, hq, length, dim]
      key:   [1, hkv, length, dim]
      value: [1, hkv, length, dim]
      mask:  [1 * hkv, 1, 1, length]
    """
    batch, hq, length, dim = query.shape
    hkv = key.shape[1]
    g = hq // hkv
    q = query.reshape(batch, hkv, g, length, dim).reshape(batch * hkv, g, length, dim)
    k = key.reshape(batch * hkv, 1, length, dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    batch, hkv, length, dim = value.shape
    g = scores.shape[1]
    v = value.reshape(batch * hkv, 1, length, dim)
    scores = scores + mask
    scores_max = torch.amax(scores, dim=-1, keepdim=True)
    probs = torch.exp(scores - scores_max)
    out = torch.matmul(probs, v) / probs.sum(dim=-1, keepdim=True)
    return out.reshape(batch, hkv * g, length, dim)


@torch.compile()
def _packed_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Legacy packed attention (kept for reference/test compatibility)."""
    return _b1_masked_attention(query, key, value, mask, scale)


def _b1_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    enable_gqa: bool,
    head_size_padded: int,
    head_size: int,
) -> torch.Tensor:
    """B=1 ``T==L`` full-bucket attention. Dest/unpack/mask unused. Compiled permute+SDPA."""
    query = _pad_head_dim_to_stick(query, head_size_padded)
    key = _pad_head_dim_to_stick(key, head_size_padded)
    value = _pad_head_dim_to_stick(value, head_size_padded)
    kernel = _compile_if_spyre(
        _b1_sdpa_kernel_gqa if enable_gqa else _b1_sdpa_kernel,
        query.device.type,
    )
    result = _call_kernel("B=1 fused encoder SDPA", kernel, query, key, value, scale)
    if result.shape[-1] == head_size:
        return result
    if result.device.type == "spyre":
        result = convert(result, "cpu")
    return result[..., :head_size].contiguous()


def _b1_seq_attention_staged(
    q_full: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    q_start: int,
    real_len: int,
    aligned_len: int,
    key_pad_mask: torch.Tensor | None,
    scale: float,
    head_size_padded: int,
    head_size: int,
    q_staging: torch.Tensor,
    k_staging: torch.Tensor,
    v_staging: torch.Tensor,
) -> torch.Tensor:
    """Execute B=1 attention on a pre-allocated device staging scratchpad.

    Copies `real_len` rows from `(q_full, k_full, v_full)` into `(q_staging, k_staging, v_staging)`,
    zeros out any padding tokens up to `aligned_len`, executes the compiled SDPA/masked kernel,
    and returns a clean `[real_len, H, head_size]` tensor with 0 host-device transfers.
    """
    hq = q_full.shape[1]
    hkv = k_full.shape[1]

    # Staging slice: [aligned_len, H, head_size_padded]
    q_pad = q_staging[:aligned_len]
    k_pad = k_staging[:aligned_len]
    v_pad = v_staging[:aligned_len]

    # Copy real tokens in place on device
    q_pad[:real_len, :, :head_size].copy_(q_full[q_start : q_start + real_len])
    k_pad[:real_len, :, :head_size].copy_(k_full[q_start : q_start + real_len])
    v_pad[:real_len, :, :head_size].copy_(v_full[q_start : q_start + real_len])

    # Zero pad tokens up to aligned_len
    if aligned_len > real_len:
        q_pad[real_len:aligned_len].zero_()
        k_pad[real_len:aligned_len].zero_()
        v_pad[real_len:aligned_len].zero_()

    # Zero pad head dims if head_size_padded > head_size (e.g. MiniLM D=32)
    if head_size_padded > head_size:
        q_pad[:, :, head_size:head_size_padded].zero_()
        k_pad[:, :, head_size:head_size_padded].zero_()
        v_pad[:, :, head_size:head_size_padded].zero_()

    if key_pad_mask is None:
        kernel = _compile_if_spyre(
            _b1_sdpa_kernel_gqa if hq != hkv else _b1_sdpa_kernel,
            q_full.device.type,
        )
        out = _call_kernel("B=1 fused encoder SDPA", kernel, q_pad, k_pad, v_pad, scale)
    else:
        # 4D view: [1, H, L, D]
        q_4d = q_pad.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        k_4d = k_pad.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        v_4d = v_pad.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        kernel = _compile_if_spyre(_b1_masked_attention, q_full.device.type)
        out_4d = _call_kernel(
            "B=1 masked encoder SDPA",
            kernel,
            q_4d,
            k_4d,
            v_4d,
            key_pad_mask,
            scale,
        )
        out = out_4d.squeeze(0).permute(1, 0, 2).contiguous()

    if out.shape[-1] != head_size:
        if out.device.type == "spyre":
            out = convert(out, "cpu")
        out = out[..., :head_size].contiguous()

    return out[:real_len]


def _index_copy(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Eager on CPU; compiled ``index_copy_`` on Spyre (eager falls back / rejects)."""
    compiled = _compile_if_spyre(_index_copy_kernel, dst.device.type)
    return compiled(dst, index, src)


def _dest_on_flat_device(dest_idx: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    """Spyre compiled index_copy_ wants int32; eager CPU wants int64."""
    if dest_idx.device.type == "spyre":
        return dest_idx
    if flat.device.type == "spyre":
        return convert(dest_idx.to(torch.int32), flat.device)
    return dest_idx.to(device=flat.device, dtype=torch.int64)


def _zeros_slot_major(
    rows: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """``[rows, H, D]`` zeros. Spyre pins rows at device position 0.

    ``empty`` + ``zero_`` stays on device. Pass size as one tuple —
    ``empty_with_layout`` rejects ``empty(B, L, H, D, device_layout=)``.
    """
    if device.type != "spyre":
        return torch.zeros(rows, num_heads, head_size, dtype=dtype, device=device)
    layout = slot_major_kv_layout(rows, num_heads, head_size, dtype)
    return torch.empty(  # ty: ignore[no-matching-overload]
        (rows, num_heads, head_size),
        dtype=dtype,
        device=device,
        device_layout=layout,
    ).zero_()


def _cached_encoder_workspace(
    attn_metadata: SpyreAttentionMetadata,
    attr: str,
    rows: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Reuse the step's slot-major scratch; allocate on first pack."""
    workspace = getattr(attn_metadata, attr)
    if (
        workspace is not None
        and workspace.shape == (rows, num_heads, head_size)
        and workspace.dtype == dtype
        and workspace.device.type == device.type
    ):
        return workspace
    workspace = _zeros_slot_major(rows, num_heads, head_size, dtype, device)
    setattr(attn_metadata, attr, workspace)
    return workspace


def gather_pack(
    flat: torch.Tensor,
    pack_indices: torch.Tensor,
    head_size_padded: int,
) -> torch.Tensor:
    """Pack via ``F.pad`` zero row + ``index_select`` of ``B×L``.

    Reference / tests. Serve uses ``scatter_pack``. ``B=1`` identity skips
    ``index_select``.
    """
    batch, aligned_len = pack_indices.shape
    _t, num_heads, _d = flat.shape
    flat = _pad_head_dim_to_stick(flat, head_size_padded)
    if batch == 1 and _is_identity_row_map(pack_indices, flat.shape[0]):
        packed = flat.unsqueeze(0)
        return packed.permute(0, 2, 1, 3).contiguous()
    flat_ext = F.pad(flat, (0, 0, 0, 0, 0, 1))
    gathered = select_rows(flat_ext, pack_indices)  # [B*L, H, Dp]
    packed = gathered.view(batch, aligned_len, num_heads, head_size_padded)
    return packed.permute(0, 2, 1, 3).contiguous()


def scatter_pack(
    flat: torch.Tensor,
    dest_idx: torch.Tensor,
    batch: int,
    aligned_len: int,
    head_size_padded: int,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack varlen ``[T, H, D]`` → ``[B, H, L, Dp]`` via compiled ``index_copy_``.

    ``dest_idx`` is ``[T]`` packed-row ids (host or device). Body-pad dests
    write an extra dummy row (``B×L``), not CLS.

    ``B=1`` with ``T == L`` skips ``index_copy_``: the runner already padded
    the body to the SDPA length. The attention mask hides leftover pad tokens.

    Spyre workspace is slot-major so ``view(B, L, …)`` splits an outermost
    dim (decoder KV). Default-layout ``view`` after ``index_copy_`` scrambles
    ``B>1``. Serve caches ``workspace`` on the step; stale pad-slot values
    from the previous pack are harmless because the key-pad mask makes K pad
    columns contribute zero probability, Q pad rows are never gathered by
    unpack, and dummy-sequence rows (batch_bucket > num_seqs) are also never
    gathered.

    ``permute.contiguous`` does **not** copy when ``H == 1`` (size-1 dim is
    ignored by ``is_contiguous``). If the result still aliases ``workspace``,
    clone so a later pack into the same buffer cannot clobber this tensor.
    """
    _t, num_heads, _d = flat.shape
    flat = _pad_head_dim_to_stick(flat, head_size_padded)
    # Fused QKV views are strided (BGE ``stride=(2304, 64, 1)``). Compiled
    # ``index_copy_`` from that layout writes the wrong rows.
    if not flat.is_contiguous():
        flat = flat.contiguous()
    if _is_b1_dense_body(batch, flat.shape[0], aligned_len):
        packed = flat.unsqueeze(0)
        return packed.permute(0, 2, 1, 3).contiguous()
    packed_rows = batch * aligned_len
    rows = packed_rows + 1
    # Extra row is the dummy dest. Slot-major prefix view is 2c on hardware.
    if (
        workspace is None
        or workspace.shape != (rows, num_heads, head_size_padded)
        or workspace.dtype != flat.dtype
        or workspace.device.type != flat.device.type
    ):
        workspace = _zeros_slot_major(
            rows,
            num_heads,
            head_size_padded,
            flat.dtype,
            flat.device,
        )
    _index_copy(workspace, _dest_on_flat_device(dest_idx, flat), flat)
    packed = workspace[:packed_rows].view(batch, aligned_len, num_heads, head_size_padded)
    packed = packed.permute(0, 2, 1, 3).contiguous()
    if packed.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr():
        packed = packed.clone()
    return packed


def reachable_pack_shapes(
    cells: list[tuple[int, int]],
    body_buckets: list[int],
    max_num_batched_tokens: int,
) -> list[tuple[int, int, int]]:
    """``(batch, aligned_len, num_src)`` triples the pack kernel can be called with.

    ``_index_copy_kernel`` guards on dest rows (``B*L + 1``) *and* source rows (the
    body token bucket). Those axes come from different bucketers and vary
    independently, so a warmed cell reached at a new token count still compiles.

    Source rows cap at the bucket covering ``min(budget, B*L)`` -- every sequence in a
    cell fits ``L`` -- but have no lower bound, because ``L`` is the smallest *warmed*
    length: three 134-token sequences land on ``(3, 320)`` with a 256-row body.

    Only what ``_is_b1_dense_body`` skips is dropped, which is narrower than all of
    ``B == 1``: the ladders can disagree at the top (``max_model_len`` 320 vs a 512
    budget), leaving one sequence with more body rows than its own length bucket.
    """
    triples: set[tuple[int, int, int]] = set()
    for batch, aligned_len in cells:
        widest = next_bucket(min(max_num_batched_tokens, batch * aligned_len), body_buckets)
        for num_src in body_buckets:
            if num_src <= widest and not _is_b1_dense_body(batch, num_src, aligned_len):
                triples.add((batch, aligned_len, num_src))
    return sorted(triples)


def gather_unpack(
    attn_out: torch.Tensor,
    unpack_indices: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Unpack padded ``[B, H, L, Dp]`` → flat ``[T, H, D]`` via ``index_select``.

    Identity ``B=1`` (``T == B×L``) is a reshape; pad / multi-seq still gather.
    """
    batch, num_heads, aligned_len, head_size_padded = attn_out.shape
    tokens = attn_out.permute(0, 2, 1, 3).contiguous()
    flat_padded = tokens.reshape(batch * aligned_len, num_heads, head_size_padded)
    if _is_identity_row_map(unpack_indices, flat_padded.shape[0]) or _is_b1_dense_body(
        batch, unpack_indices.shape[0], aligned_len
    ):
        gathered = flat_padded
    else:
        gathered = select_rows(flat_padded, unpack_indices)
    if gathered.shape[-1] == head_size:
        return gathered
    if gathered.device.type == "spyre":
        gathered = convert(gathered, "cpu")
    return gathered[..., :head_size].contiguous()


def _indices_for_device(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move pack dest / unpack indices onto ``device`` once.

    Spyre ``index_select`` and compiled ``index_copy_`` want int32; CPU wants int64.
    """
    if device.type == "spyre":
        cpu = indices if indices.device.type == "cpu" else indices.cpu()
        return convert(cpu.to(torch.int32), device)
    return indices.to(device=device, dtype=torch.long)


def _ladder_encoder_shape(
    num_seqs: int,
    max_len: int,
    max_model_len: int,
) -> tuple[int, int]:
    """Snap a batch no warmed cell covers onto the warmup length ladder.

    ``pick_encoder_attention_shape`` misses routinely, not exceptionally: warmup drops
    every cell with ``B*L > encoder_cell_budget(...)``, so two 300-token prompts have no
    warmed ``(2, 512)`` to land on. Stick-aligning ``max_len`` instead would emit
    non-buckets (320, 448, ...), one compile per distinct prompt length.

    The batch stays exact. Rounding it up cannot help -- the caller already searched
    every warmed cell -- and costs, since scores are ``[B*Hkv, G, L, L]``: one 400-token
    request with eight short ones would round to ``(16, 512)``, mostly padding.
    """
    batch = num_seqs
    length = next_bucket(max_len, default_encoder_len_buckets(max_model_len))
    # Warmup's own body runs miss too (max_num_seqs seqs of size//B tokens) and
    # compiling those is the point, so only a serving-path miss is news.
    #
    # info, not warning: pooling_warmup_shapes warms a finite set, so a batch wider
    # than it takes this path normally. _call_kernel reports the compile itself.
    if is_warmup_complete():
        # Args are part of info_once's dedup key, so they must be the ladder
        # cell and not the request: num_seqs x max_len has thousands of values.
        logger.info_once(
            "Encoder attention fell back to ladder shape (B=%d, L=%d): no warmed cell covers "
            "the batch, so this compiles on first use. Warmup covers B*L up to "
            "encoder_cell_budget -- see pooling_warmup_shapes.",
            batch,
            length,
        )
    return batch, length


def _ensure_encoder_pack(
    attn_metadata: SpyreAttentionMetadata,
    *,
    padded_tokens: int,
    n: int,
    query: torch.Tensor,
    num_kv_heads: int,
    target_device: torch.device,
    cached_encoder_shapes: list[tuple[int, int]],
    cached_max_num_seqs: int,
    cached_max_model_len: int,
    cached_max_num_batched_tokens: int,
) -> None:
    """Build per-sequence metadata and key-pad masks once per step."""
    if attn_metadata.encoder_pack_batch is not None:
        return

    qsl = attn_metadata.query_start_loc.cpu()
    q_starts = qsl[:-1].tolist()
    qsl_lens = torch.diff(qsl).tolist()
    kv_lens = attn_metadata.seq_lens.cpu().tolist()
    num_seqs = attn_metadata.num_seqs
    q_starts = q_starts[:num_seqs]
    qsl_lens = qsl_lens[:num_seqs]
    kv_lens = kv_lens[:num_seqs]

    query_lens = _content_query_lens(qsl_lens, kv_lens, num_actual_tokens=n)
    max_len = max(query_lens, default=0)
    pair = pick_encoder_attention_shape(
        num_seqs,
        max_len,
        cached_encoder_shapes,
        cached_max_num_seqs,
        cached_max_model_len,
        cached_max_num_batched_tokens,
    )
    if pair is not None:
        batch_bucket, aligned_len = pair
    else:
        batch_bucket, aligned_len = _ladder_encoder_shape(num_seqs, max_len, cached_max_model_len)

    fused = _is_b1_fused_sdpa(
        batch_bucket,
        padded_tokens,
        aligned_len,
        query_lens[0] if query_lens else 0,
    )
    if fused:
        attn_metadata.encoder_pack_batch = batch_bucket
        attn_metadata.encoder_pack_len = aligned_len
        attn_metadata.encoder_fused_sdpa = True
        return

    # Build per-sequence metadata with warmed length buckets
    len_buckets_list = default_encoder_len_buckets(cached_max_model_len)
    seq_info = []
    for s in range(num_seqs):
        q_start = q_starts[s]
        real_len = query_lens[s]
        seq_bucket_len = next_bucket(real_len, len_buckets_list)
        if real_len == seq_bucket_len:
            seq_key_pad = None
        else:
            mask_cpu = build_attention_mask(
                1,
                seq_bucket_len,
                [real_len],
                [real_len],
                dtype=query.dtype,
                device=torch.device("cpu"),
            )
            seq_key_pad = host_key_pad_mask(mask_cpu, num_kv_heads)
            if target_device.type == "spyre":
                seq_key_pad = convert(seq_key_pad, target_device)
            else:
                seq_key_pad = seq_key_pad.to(target_device)
        seq_info.append((q_start, real_len, seq_bucket_len, seq_key_pad))

    attn_metadata.encoder_seq_info = seq_info
    # Keep encoder_key_pad_mask and legacy attributes for compatibility/tests
    if len(seq_info) == 1:
        attn_metadata.encoder_key_pad_mask = seq_info[0][3]
    attn_metadata.encoder_pack_batch = batch_bucket
    attn_metadata.encoder_pack_len = aligned_len
    attn_metadata.encoder_fused_sdpa = False


def build_attention_mask(
    num_seqs: int,
    aligned_len: int,
    query_lens: list[int],
    kv_lens: list[int],
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Additive mask ``[B, 1, L, L]``: 0 where attend, ``-inf`` elsewhere.

    Built on the host (vectorized ``lt`` + nested ``where``), then ``convert``'d.
    On-device materialization is not stick-safe: Spyre cannot produce bool from
    int32 ``lt``, and cannot broadcast ``where`` of ``[B,1,L,1]`` × ``[B,1,1,L]``
    into ``[B,1,L,L]`` (no stick-scatter).
    """
    if device is None:
        device = torch.device("cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if num_seqs != len(query_lens):
        raise ValueError(f"num_seqs={num_seqs} != len(query_lens)={len(query_lens)}")

    q_len = torch.tensor(query_lens, dtype=torch.int32)
    kv_len = torch.tensor(
        [min(q, k) for q, k in zip(query_lens, kv_lens)],
        dtype=torch.int32,
    )
    q_pos = torch.arange(aligned_len, dtype=torch.int32)
    kv_pos = torch.arange(aligned_len, dtype=torch.int32)
    zeros = torch.zeros((), dtype=dtype)
    neg_inf = torch.tensor(torch.finfo(dtype).min, dtype=dtype)

    q_ok = (q_pos.unsqueeze(0) < q_len.unsqueeze(1)).unsqueeze(1).unsqueeze(-1)
    k_ok = (kv_pos.unsqueeze(0) < kv_len.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
    mask = torch.where(q_ok, torch.where(k_ok, zeros, neg_inf), neg_inf)
    if device.type == "spyre":
        return convert(mask, device)
    return mask.to(device)


class SpyreEncoderAttentionImpl(SpyreAttentionImpl):
    """Bidirectional encoder self-attention (no KV cache).

    The platform selects this impl for ENCODER/ENCODER_ONLY layers (see
    ``TorchSpyrePlatform.get_attn_backend_cls``), so there is no per-call
    ``attn_type`` branch. Setup is shared with the paged decoder impl; forward
    packs Q/K/V with scatter into a slot-major workspace (decoder KV
    layout; extra dummy row for body-pad), then attention and gather-unpack.
    ``B=1`` ``T == L`` with a full prompt compiles permute+SDPA; live pad
    uses packed QK (matmul and P·V compiled apart so Inductor cannot fuse
    to ``F.sdpa``).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # get_current_vllm_config() only works at construction time; forward()
        # runs through a custom-op boundary that loses the context.
        cfg = get_current_vllm_config()
        self._cached_max_num_seqs = cfg.scheduler_config.max_num_seqs
        self._cached_max_model_len = cfg.model_config.max_model_len
        self._cached_max_num_batched_tokens = cfg.scheduler_config.max_num_batched_tokens
        self._cached_encoder_shapes = pooling_warmup_shapes(
            max_num_seqs=self._cached_max_num_seqs,
            max_model_len=self._cached_max_model_len,
            max_num_batched_tokens=self._cached_max_num_batched_tokens,
            len_bucket=default_encoder_len_buckets(self._cached_max_model_len),
        )
        # The runner's 1D body ladder, the pack kernel's second shape axis. None under
        # enforce_eager, where the platform hook never builds it and record_pack_graphs
        # has nothing to trace.
        body_buckets = cfg.compilation_config.compile_sizes or ()
        self._cached_body_buckets = [int(size) for size in body_buckets]

    def record_graphs(self, *args, **kwargs) -> int:
        """Nothing to page: ``forward`` packs Q/K/V, and its shapes are warmed by the
        runner's ``_warmup_pooling_bucket_shapes`` plus ``record_pack_graphs``."""
        return 0

    def forward(  # ty: ignore[invalid-method-override]
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: SpyrePagedKVCache,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,  # [num_tokens, num_heads, head_size]
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, kv_cache, output_scale, output_block_scale
        if attn_metadata is None:
            return output

        # query/key/value/output are padded to the runner's warmed body-bucket
        # size, not num_actual_tokens. Keep that shape or index_select
        # recompiles per request.
        n = attn_metadata.num_actual_tokens
        padded_tokens = query.shape[0]
        target_device = output.device
        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        head_size = query.shape[2]
        head_size_padded = _align_up(head_size)
        scale = self.scale

        _ensure_encoder_pack(
            attn_metadata,
            padded_tokens=padded_tokens,
            n=n,
            query=query,
            num_kv_heads=num_kv_heads,
            target_device=target_device,
            cached_encoder_shapes=self._cached_encoder_shapes,
            cached_max_num_seqs=self._cached_max_num_seqs,
            cached_max_model_len=self._cached_max_model_len,
            cached_max_num_batched_tokens=self._cached_max_num_batched_tokens,
        )

        if query.device.type != target_device.type:
            query = convert(query, target_device.type)
            key = convert(key, target_device.type)
            value = convert(value, target_device.type)

        if attn_metadata.encoder_fused_sdpa:
            result = _b1_dense_attention(
                query,
                key,
                value,
                scale,
                num_kv_heads != num_heads,
                head_size_padded,
                head_size,
            )
            if result.dtype != output.dtype:
                result = convert(result, dtype=output.dtype)

            # MiniLM D=32: flatten to [T, H*D] (384 = 6 sticks) so the write is aligned.
            use_flat_write = target_device.type == "spyre" and head_size % ENCODER_SEQ_ALIGNMENT != 0
            if use_flat_write:
                if result.device.type == "spyre":
                    result = convert(result, "cpu")
                src = convert(
                    result.reshape(padded_tokens, -1).contiguous(), target_device.type, output.dtype
                )
                output.reshape(padded_tokens, -1).copy_(src)
            else:
                if result.device.type != output.device.type:
                    result = convert(result, output.device)
                output.copy_(result)
            return output

        seq_info = attn_metadata.encoder_seq_info
        assert seq_info is not None

        # Pre-allocated device staging buffers (allocated once on attn_metadata, 0 dynamic allocations)
        q_ws = _cached_encoder_workspace(
            attn_metadata,
            "encoder_q_workspace",
            self._cached_max_model_len,
            num_heads,
            head_size_padded,
            query.dtype,
            query.device,
        )
        k_ws = _cached_encoder_workspace(
            attn_metadata,
            "encoder_kv_workspace",
            self._cached_max_model_len,
            num_kv_heads,
            head_size_padded,
            key.dtype,
            key.device,
        )
        v_ws = _cached_encoder_workspace(
            attn_metadata,
            "encoder_v_workspace",
            self._cached_max_model_len,
            num_kv_heads,
            head_size_padded,
            value.dtype,
            value.device,
        )

        use_flat_write = target_device.type == "spyre" and head_size % ENCODER_SEQ_ALIGNMENT != 0
        for q_start, real_len, seq_aligned_len, key_pad in seq_info:
            out_slice = _b1_seq_attention_staged(
                query,
                key,
                value,
                q_start,
                real_len,
                seq_aligned_len,
                key_pad,
                scale,
                head_size_padded,
                head_size,
                q_ws,
                k_ws,
                v_ws,
            )
            if out_slice.dtype != output.dtype:
                out_slice = convert(out_slice, dtype=output.dtype)

            if use_flat_write:
                if out_slice.device.type == "spyre":
                    out_slice = convert(out_slice, "cpu")
                src = convert(
                    out_slice.reshape(real_len, -1).contiguous(), target_device.type, output.dtype
                )
                output[q_start : q_start + real_len].reshape(real_len, -1).copy_(src)
            else:
                if out_slice.device.type != output.device.type:
                    out_slice = convert(out_slice, output.device)
                output[q_start : q_start + real_len].copy_(out_slice)

        return output

    def record_pack_graphs(self, device: torch.device) -> int:
        """Trace B=1 attention kernels on all reachable length buckets.

        Loops through each length bucket in default_encoder_len_buckets and records
        both dense SDPA and masked SDPA.
        """
        if not self._compile_attn or device.type != "spyre":
            return 0
        lengths = default_encoder_len_buckets(self._cached_max_model_len)
        head_size_padded = _align_up(self.head_size)
        recorded = 0
        for length in lengths:
            q_padded = convert(
                torch.zeros(length, self.num_heads, head_size_padded, dtype=self.model_dtype),
                device,
            )
            k_padded = convert(
                torch.zeros(length, self.num_kv_heads, head_size_padded, dtype=self.model_dtype),
                device,
            )
            v_padded = convert(
                torch.zeros(length, self.num_kv_heads, head_size_padded, dtype=self.model_dtype),
                device,
            )
            # Trace dense SDPA
            try:
                kernel = _compile_if_spyre(
                    _b1_sdpa_kernel_gqa if self.num_heads != self.num_kv_heads else _b1_sdpa_kernel,
                    device.type,
                )
                _call_kernel("B=1 fused encoder SDPA", kernel, q_padded, k_padded, v_padded, self.scale)
                recorded += 1
            except Exception:
                logger.warning(
                    "B=1 dense encoder SDPA (L=%d) failed to record.",
                    length,
                    exc_info=True,
                )

            # Trace masked SDPA
            try:
                mask_cpu = build_attention_mask(
                    1, length, [length // 2 or 1], [length // 2 or 1], dtype=self.model_dtype, device="cpu"
                )
                key_pad = host_key_pad_mask(mask_cpu, self.num_kv_heads)
                key_pad = convert(key_pad, device)
                q_4d = q_padded.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                k_4d = k_padded.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                v_4d = v_padded.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                kernel_masked = _compile_if_spyre(_b1_masked_attention, device.type)
                _call_kernel(
                    "B=1 masked encoder SDPA",
                    kernel_masked,
                    q_4d,
                    k_4d,
                    v_4d,
                    key_pad,
                    self.scale,
                )
                recorded += 1
            except Exception:
                logger.warning(
                    "B=1 masked encoder SDPA (L=%d) failed to record.",
                    length,
                    exc_info=True,
                )
        return recorded


class SpyreEncoderAttentionBackend(SpyreAttentionBackend):
    """Encoder-only (no KV cache) variant of the Spyre backend."""

    # These layers have no KV cache, but vLLM still hands encoder-only specs a
    # zero-filled slot mapping, so upstream must skip `unified_kv_cache_update` entirely.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_impl_cls() -> type[SpyreEncoderAttentionImpl]:
        return SpyreEncoderAttentionImpl
