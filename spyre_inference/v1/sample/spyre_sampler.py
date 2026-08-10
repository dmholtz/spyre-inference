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


import torch

from vllm.v1.sample.ops.top_k_top_p_sampler import TopKTopPSampler, apply_top_k_top_p

from spyre_inference.v1.sample.async_ring_buffer import AsyncExponential_RingBuffer

def sample_with_predrawn_noise(
    probs: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    """Sample using pre-drawn exponential noise (no exponential_() call)."""
    return probs.div(noise).argmax(dim=-1).view(-1)

class SpyreTopKTopPSampler(TopKTopPSampler):
    """TODO"""

    @classmethod
    def from_base_instance(cls, base_sampler: TopKTopPSampler) -> "SpyreTopKTopPSampler":
        """Construct a Spyre sampler from a base TopKTopPSampler instance.

        Args:
            base_sampler: The base TopKTopPSampler instance to convert.
        """
        return cls(logprobs_mode=base_sampler.logprobs_mode, use_fp64_gumbel=base_sampler.use_fp64_gumbel)
        

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #
        self._noise_buffer: AsyncExponential_RingBuffer | None = None

        self.forward = self.forward_spyre

    def forward_spyre(self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        
        """
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)

        self._lazy_ring_buffer_init(vocab_size=probs.shape[1])

        with self._noise_buffer.borrow_rows(n=probs.shape[0]) as noise:
            sample_result = sample_with_predrawn_noise(probs, noise)

        return sample_result, logits_to_return

    def _lazy_ring_buffer_init(self, vocab_size: int) -> None:
        if self._noise_buffer is None:
            max_batch_size = 32
            self._noise_buffer = AsyncExponential_RingBuffer(
                vocab_size, max_batch_size
            )
