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

from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.sample.sampler import Sampler, SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from spyre_inference.v1.sample.rejection_sampler import (
    SpyreRejectionSampler,
    sample_recovered_tokens_fused,
    spyre_rejection_sample,
)
from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels


@pytest.fixture(autouse=True, scope="module")
def _kernel_overrides():
    """Install the PyTorch rejection kernels, as WorkerBase does at worker init.

    Without Triton the stock kernels are undecorated functions that vLLM's launch
    wrapper cannot bind, so the sampler fails before reaching any Spyre code.
    """
    from spyre_inference.platform import TorchSpyrePlatform

    TorchSpyrePlatform.register_triton_kernel_overrides()


def _make_sampling_metadata(
    batch_size: int,
    all_greedy: bool = True,
    generators: dict[int, torch.Generator] | None = None,
) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.zeros(batch_size) if all_greedy else torch.full((batch_size,), 0.7),
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=None,
        top_k=None,
        generators=generators or {},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch_size),
        presence_penalties=torch.zeros(batch_size),
        repetition_penalties=torch.ones(batch_size),
        output_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=None,
    )


def test_greedy_rejection_sample_all_accepted():
    draft_tokens = torch.tensor([10, 20, 30], dtype=torch.int64)
    num_draft_tokens = [3]
    cu_num_draft_tokens = torch.tensor([3], dtype=torch.int32)
    bonus_token_ids = torch.tensor([40], dtype=torch.int64)

    target_logits = torch.full((3, 100), -10.0)
    target_logits[0, 10] = 10.0
    target_logits[1, 20] = 10.0
    target_logits[2, 30] = 10.0

    meta = _make_sampling_metadata(batch_size=1, all_greedy=True)

    out = spyre_rejection_sample(
        draft_token_ids=draft_tokens,
        num_draft_tokens=num_draft_tokens,
        max_spec_len=3,
        cu_num_draft_tokens=cu_num_draft_tokens,
        draft_probs=None,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=meta,
    )

    # All accepted -> [10, 20, 30, bonus=40]
    assert out.tolist() == [[10, 20, 30, 40]]


def test_greedy_rejection_sample_partial_rejection():
    draft_tokens = torch.tensor([10, 20, 30], dtype=torch.int64)
    num_draft_tokens = [3]
    cu_num_draft_tokens = torch.tensor([3], dtype=torch.int32)
    bonus_token_ids = torch.tensor([40], dtype=torch.int64)

    target_logits = torch.full((3, 100), -10.0)
    target_logits[0, 10] = 10.0
    target_logits[1, 25] = 10.0  # rejects token 1 (was 20, target says 25)
    target_logits[2, 30] = 10.0

    meta = _make_sampling_metadata(batch_size=1, all_greedy=True)

    out = spyre_rejection_sample(
        draft_token_ids=draft_tokens,
        num_draft_tokens=num_draft_tokens,
        max_spec_len=3,
        cu_num_draft_tokens=cu_num_draft_tokens,
        draft_probs=None,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=meta,
    )

    # Token 0 accepted (10), token 1 rejected & replaced by target argmax (25), rest placeholder
    assert out.tolist() == [[10, 25, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]]


def test_sample_recovered_tokens_fused_distribution():
    if not has_sampling_kernels():
        pytest.skip("Sampling kernels extension not available")

    vocab_size = 50
    target_probs = torch.zeros((1, vocab_size), dtype=torch.float32)
    # Target distribution puts prob on tokens 5, 10, 15
    target_probs[0, 5] = 0.5
    target_probs[0, 10] = 0.3
    target_probs[0, 15] = 0.2

    draft_probs = torch.zeros((1, vocab_size), dtype=torch.float32)
    draft_probs[0, 5] = 0.5  # Completely cancels token 5

    draft_token_ids = torch.tensor([5], dtype=torch.int64)
    cu_num_draft_tokens = torch.tensor([1], dtype=torch.int32)
    num_draft_tokens = [1]

    # Sample many times and check that only token 10 and 15 are drawn (token 5 is 0 residual)
    counts = {10: 0, 15: 0}
    draws = 500
    for seed in range(draws):
        gen = torch.Generator().manual_seed(seed)
        meta = _make_sampling_metadata(batch_size=1, all_greedy=False, generators={0: gen})
        rec = sample_recovered_tokens_fused(
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens,
            draft_token_ids=draft_token_ids,
            draft_probs=draft_probs,
            target_probs=target_probs,
            sampling_metadata=meta,
        )
        tok = int(rec[0].item())
        assert tok in counts, f"Unexpected token {tok} sampled with residual prob 0"
        counts[tok] += 1

    # Token 10 (residual prob 0.3) should be drawn roughly 60% of the time (0.3 / 0.5)
    # Token 15 (residual prob 0.2) roughly 40% of the time (0.2 / 0.5)
    p10 = counts[10] / draws
    assert 0.45 <= p10 <= 0.75


def test_spyre_rejection_sampler_forward_no_exponential_calls():
    mock_sampler = Mock(spec=Sampler)
    mock_sampler.logprobs_mode = "raw_logits"
    mock_sampler.use_fp64_gumbel = False
    mock_sampler.return_value = SamplerOutput(
        sampled_token_ids=torch.tensor([40], dtype=torch.int64),
        logprobs_tensors=None,
    )

    spec_config = SpeculativeConfig(
        num_speculative_tokens=2,
    )
    sampler = SpyreRejectionSampler(
        mock_sampler,
        spec_config=spec_config,
        device=torch.device("cpu"),
    )

    metadata = SpecDecodeMetadata.make_dummy([[10, 20]], device=torch.device("cpu"))
    metadata.target_logits_indices = torch.tensor([0, 1], dtype=torch.int64)
    metadata.bonus_logits_indices = torch.tensor([2], dtype=torch.int64)

    logits = torch.randn(3, 100, dtype=torch.float32)
    meta = _make_sampling_metadata(batch_size=1, all_greedy=False)

    with patch.object(torch.Tensor, "exponential_", wraps=torch.Tensor.exponential_) as mock_exp:
        out = sampler(
            metadata=metadata,
            draft_probs=None,
            logits=logits,
            sampling_metadata=meta,
        )
        assert out.sampled_token_ids is not None
        if has_sampling_kernels():
            # Crucial assertion: aten::exponential_ was NEVER invoked!
            mock_exp.assert_not_called()


# Padded drafts: TorchSpyreScheduler pads every draft list to k with -1, which the
# rejection sampler must always reject without disturbing the real drafts before it.

PAD = -1


def test_greedy_kernel_rejects_padded_draft():
    from spyre_inference.v1.worker.spyre_model_runner import (
        _spyre_rejection_greedy_sample_kernel_impl,
    )

    out = torch.full((1, 4), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    _spyre_rejection_greedy_sample_kernel_impl(
        output_token_ids_ptr=out,
        cu_num_draft_tokens_ptr=torch.tensor([3]),
        draft_token_ids_ptr=torch.tensor([10, PAD, PAD]),
        target_argmax_ptr=torch.tensor([10, 20, 30]),
        bonus_token_ids_ptr=torch.tensor([40]),
        is_greedy_ptr=None,
        max_spec_len=3,
    )
    # The real draft is accepted; the first pad is replaced by the target's own token,
    # which is what the bonus slot would have held without padding.
    assert out.tolist() == [[10, 20, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]]


def test_random_kernel_rejects_padded_draft_even_at_zero_uniform():
    from spyre_inference.v1.worker.spyre_model_runner import (
        _spyre_rejection_random_sample_kernel_impl,
    )

    vocab = 8
    out = torch.full((1, 3), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    _spyre_rejection_random_sample_kernel_impl(
        output_token_ids_ptr=out,
        cu_num_draft_tokens_ptr=torch.tensor([2]),
        draft_token_ids_ptr=torch.tensor([3, PAD]),
        draft_probs_ptr=None,
        target_probs_ptr=torch.full((2, vocab), 1.0 / vocab),
        bonus_token_ids_ptr=torch.tensor([7]),
        recovered_token_ids_ptr=torch.tensor([0, 5]),
        # 0.0 accepts any real draft: only the pad check can reject position 1.
        uniform_probs_ptr=torch.zeros(2, dtype=torch.float64),
        is_greedy_ptr=torch.tensor([False]),
        max_spec_len=2,
        vocab_size=vocab,
        NO_DRAFT_PROBS=True,
    )
    assert out.tolist() == [[3, 5, PLACEHOLDER_TOKEN_ID]]


def test_recovered_fallback_keeps_last_vocab_id_for_padded_draft():
    """A pad proposed nothing, so the residual is the whole target distribution."""
    from spyre_inference.v1.worker.spyre_model_runner import (
        _spyre_sample_recovered_tokens_kernel_impl,
    )

    vocab = 8
    target = torch.full((1, vocab), 0.01)
    target[0, vocab - 1] = 0.93  # the last id, which indexing with -1 would hit
    out = torch.empty(1, dtype=torch.int64)
    _spyre_sample_recovered_tokens_kernel_impl(
        output_token_ids_ptr=out,
        cu_num_draft_tokens_ptr=torch.tensor([1]),
        draft_token_ids_ptr=torch.tensor([PAD]),
        draft_probs_ptr=None,
        target_probs_ptr=target,
        inv_q_ptr=torch.ones(1, vocab),
        vocab_size=vocab,
        NO_DRAFT_PROBS=True,
    )
    assert out.item() == vocab - 1


def test_recovered_fallback_still_excludes_a_real_draft():
    from spyre_inference.v1.worker.spyre_model_runner import (
        _spyre_sample_recovered_tokens_kernel_impl,
    )

    vocab = 8
    target = torch.full((1, vocab), 0.01)
    target[0, 2] = 0.9
    target[0, 6] = 0.03
    out = torch.empty(1, dtype=torch.int64)
    _spyre_sample_recovered_tokens_kernel_impl(
        output_token_ids_ptr=out,
        cu_num_draft_tokens_ptr=torch.tensor([1]),
        draft_token_ids_ptr=torch.tensor([2]),
        draft_probs_ptr=None,
        target_probs_ptr=target,
        inv_q_ptr=torch.ones(1, vocab),
        vocab_size=vocab,
        NO_DRAFT_PROBS=True,
    )
    assert out.item() == 6


def test_greedy_rejection_sample_padded_verification():
    """Two requests at k=3: one fully drafted, one with a single real draft."""
    target_logits = torch.full((6, 100), -10.0)
    for row, tok in enumerate([10, 20, 30, 11, 21, 31]):
        target_logits[row, tok] = 10.0

    out = spyre_rejection_sample(
        draft_token_ids=torch.tensor([10, 20, 30, 11, PAD, PAD]),
        num_draft_tokens=[3, 3],
        max_spec_len=3,
        cu_num_draft_tokens=torch.tensor([3, 6], dtype=torch.int32),
        draft_probs=None,
        target_logits=target_logits,
        bonus_token_ids=torch.tensor([40, 41]),
        sampling_metadata=_make_sampling_metadata(batch_size=2, all_greedy=True),
    )
    assert out.tolist() == [
        [10, 20, 30, 40],
        [11, 21, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
    ]
