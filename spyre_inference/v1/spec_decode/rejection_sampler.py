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

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.32.0 dispatcher adapters for the vendored csrc/cpu/spec_decode_utils.cpp.

Adapted from vllm/utils/cpu_triton_utils.py at 5df63a7203. The native kernels
accept the caller's dtypes directly and implement synthetic acceptance.

``install`` additionally swaps upstream's recovered-token sampling for one that draws
its noise in-kernel; the rest of vLLM's RejectionSampler runs unchanged.
"""

import torch
from vllm.v1.sample import rejection_sampler as upstream
from vllm.v1.sample.metadata import SamplingMetadata

from spyre_inference.v1.sample.sampling_kernels import draw_seeds, has_sampling_kernels

_HAS_REJECTION_KERNELS = has_sampling_kernels() and all(
    hasattr(torch.ops._spyre_C, name)
    for name in (
        "rejection_greedy_sample_kernel_impl",
        "rejection_random_sample_kernel_impl",
        "expand_kernel_impl",
        "sample_recovered_tokens_kernel_impl",
        "sample_recovered_tokens_seeded_kernel_impl",
    )
)


def _kernels():
    if not _HAS_REJECTION_KERNELS:
        raise RuntimeError(
            "Rejection sampling requires the C++ kernels. "
            "Rebuild spyre-inference's csrc/ CPU extension before using speculative decoding."
        )
    return torch.ops._spyre_C


def greedy_sample(
    output,
    cumulative,
    draft_ids,
    target_ids,
    bonus_ids,
    is_greedy,
    max_spec_len,
    uniform,
    conditional_rates,
    SYNTHETIC_MODE,
):
    _kernels().rejection_greedy_sample_kernel_impl(
        output,
        cumulative,
        draft_ids,
        target_ids,
        bonus_ids,
        is_greedy,
        max_spec_len,
        uniform,
        conditional_rates,
        SYNTHETIC_MODE,
    )


def random_sample(
    output,
    cumulative,
    draft_ids,
    draft_probs,
    target_probs,
    bonus_ids,
    recovered_ids,
    uniform,
    is_greedy,
    max_spec_len,
    vocab_size,
    conditional_rates,
    NO_DRAFT_PROBS,
    SYNTHETIC_MODE,
):
    _kernels().rejection_random_sample_kernel_impl(
        output,
        cumulative,
        draft_ids,
        draft_probs,
        target_probs,
        bonus_ids,
        recovered_ids,
        uniform,
        is_greedy,
        max_spec_len,
        vocab_size,
        conditional_rates,
        NO_DRAFT_PROBS,
        SYNTHETIC_MODE,
    )


def expand(output, values, cumulative, replace_from, replace_to, MAX_NUM_TOKENS):
    _kernels().expand_kernel_impl(output, values, cumulative, replace_from, replace_to)


def recover(
    output,
    cumulative,
    draft_ids,
    draft_probs,
    target_probs,
    inverse_noise,
    vocab_size,
    BLOCK_SIZE,
    NO_DRAFT_PROBS,
    USE_FP64_GUMBEL,
):
    _kernels().sample_recovered_tokens_kernel_impl(
        output,
        cumulative,
        draft_ids,
        draft_probs,
        target_probs,
        inverse_noise,
        vocab_size,
        NO_DRAFT_PROBS,
    )


_upstream_sample_recovered_tokens = upstream.sample_recovered_tokens


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Upstream's ``sample_recovered_tokens`` without its [batch, vocab] exponential_ draw."""
    if use_fp64_gumbel:
        # The kernel's noise is fixed; keep upstream's fp64 q, as SpyreTopKTopPSampler does.
        return _upstream_sample_recovered_tokens(
            max_spec_len,
            num_draft_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            sampling_metadata,
            device,
            use_fp64_gumbel,
        )
    # Like upstream, a request without drafts leaves its generator untouched.
    generators = {
        i: generator
        for i, generator in sampling_metadata.generators.items()
        if num_draft_tokens[i] > 0
    }
    recovered_token_ids = torch.empty_like(draft_token_ids)
    _kernels().sample_recovered_tokens_seeded_kernel_impl(
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        draw_seeds(len(num_draft_tokens), generators),
        target_probs.shape[-1],
        draft_probs is None,
    )
    return recovered_token_ids


def install() -> None:
    """Route upstream ``rejection_sample`` through ``sample_recovered_tokens`` above."""
    upstream.sample_recovered_tokens = sample_recovered_tokens  # ty: ignore[invalid-assignment]
