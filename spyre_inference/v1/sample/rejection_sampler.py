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

from dataclasses import replace

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    GREEDY_TEMPERATURE,
    MAX_SPEC_LEN,
    PLACEHOLDER_TOKEN_ID,
    RejectionSampler,
    _rejection_greedy_sample,
    _rejection_random_sample,
    apply_sampling_constraints,
    generate_uniform_probs,
)
from vllm.v1.sample.rejection_sampler import (
    rejection_sample as vllm_rejection_sample,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from spyre_inference.v1.sample.sampler import greedy_sample
from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels


def sample_recovered_tokens_fused(
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Sample recovered tokens from the residual distribution using fused Gumbel-max.

    Instead of materializing [B, V] exponential noise via aten::exponential_ on CPU
    and taking reciprocal/argmax in Python, this evaluates the residual logits
    log(max(0, target_probs - draft_probs)) and samples directly using
    torch.ops._spyre_C.fused_gumbel_argmax with per-request seeds.
    """
    num_tokens = draft_token_ids.shape[0]
    batch_size = len(num_draft_tokens)
    if num_tokens == 0 or batch_size == 0:
        return torch.empty(0, dtype=torch.int64, device=draft_token_ids.device)

    # 1. Compute residual distribution in log space
    if draft_probs is None:
        residual = target_probs.clone()
        valid_mask = draft_token_ids >= 0
        if valid_mask.any():
            valid_indices = draft_token_ids[valid_mask].unsqueeze(1).to(torch.int64)
            residual[valid_mask] = residual[valid_mask].scatter(1, valid_indices, 0.0)
    else:
        residual = torch.clamp(target_probs - draft_probs, min=0.0)

    # torch.log(0.0) produces -inf, which fused_gumbel_argmax safely prunes
    residual_logits = torch.log(residual.to(torch.float32))

    # 2. Assign per-request seeds (reproducible with generators)
    batch_size = len(num_draft_tokens)
    req_seeds = torch.randint(0, 2**62, (batch_size,), dtype=torch.long)
    for i, generator in sampling_metadata.generators.items():
        if num_draft_tokens[i] > 0:
            req_seeds[i] = torch.randint(0, 2**62, (1,), generator=generator)

    # Broadcast request seeds to per-token seeds
    seeds = torch.empty(num_tokens, dtype=torch.long, device=draft_token_ids.device)
    cu = cu_num_draft_tokens.to(torch.int64)
    starts = torch.cat([cu.new_zeros(1), cu[:-1]])
    for i in range(batch_size):
        seeds[starts[i] : cu[i]] = req_seeds[i]

    # 3. Fused C++ draw (no aten::exponential, no RAM allocation)
    return torch.ops._spyre_C.fused_gumbel_argmax(residual_logits, seeds)  # ty: ignore[invalid-argument-type]


def spyre_rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Rejection sampling using Spyre's fused sampling kernels when available."""
    if not (has_sampling_kernels() and not use_fp64_gumbel):
        return vllm_rejection_sample(
            draft_token_ids,
            num_draft_tokens,
            max_spec_len,
            cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            synthetic_mode=synthetic_mode,
            synthetic_conditional_rates=synthetic_conditional_rates,
            use_fp64_gumbel=use_fp64_gumbel,
        )

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    device = target_logits.device

    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

    uniform_probs: torch.Tensor | None = None
    if synthetic_mode or not sampling_metadata.all_greedy:
        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )

    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests
        if num_tokens > 0:
            target_argmax = greedy_sample(target_logits)
        else:
            target_argmax = torch.empty(0, dtype=torch.int64, device=device)
        _rejection_greedy_sample(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
            uniform_probs,
            synthetic_conditional_rates,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Compute probability distribution from target logits
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)

    # Sample recovered tokens using fused C++ kernel
    recovered_token_ids = sample_recovered_tokens_fused(
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
    )

    # Rejection sampling for random sampling requests
    assert uniform_probs is not None
    _rejection_random_sample(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        synthetic_conditional_rates,
    )
    return output_token_ids


class SpyreRejectionSampler(RejectionSampler):
    """Rejection sampler utilizing Spyre's fused C++ sampling kernels."""

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor | None,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        raw_target_logits = logits[target_logits_indices]
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            target_logits = target_logits.clone()
        target_logits = self.apply_logits_processors(target_logits, sampling_metadata, metadata)
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        output_token_ids = spyre_rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            synthetic_mode=self.synthetic_mode,
            synthetic_conditional_rates=self.synthetic_conditional_rates,
            use_fp64_gumbel=self.use_fp64_gumbel,
        )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )
