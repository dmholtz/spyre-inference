import torch

from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p, random_sample

class SpyreTopKTopPSampler(TopKTopPSampler):
    
    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        PyTorch-native implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1).to("cpu")
        return (
            random_sample(probs, generators, self.use_fp64_gumbel),
            logits_to_return,
        )