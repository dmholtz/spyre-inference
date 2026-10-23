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

"""Benchmark of n-gram speculative decoding on Granite-4.1-8B (TP=1).

Setup: batch 1, 100 input tokens (artificial repeating 1,2,3,4 pattern),
2 speculative tokens, 2 output tokens, max_model_len 512, prefix caching off.

Instead of a blind warmup, every iteration is driven step by step through the
engine and the wall-clock time of each ``step()`` is printed. This shows whether
the warmup is complete, i.e. whether the per-step times have settled and no
step still pays a (re)compilation cost.

Usage:
    source ./setup_profile_env.sh
    export VLLM_PLUGINS=spyre_inference
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
    python benchmark_spyre_granite41_ngram_spec.py
"""

import os
import time

# external_launcher reads these from env
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

TP = 1
BATCH_SIZE = 1
INPUT_TOKENS = 100
NUM_SPECULATIVE_TOKENS = 2
OUTPUT_TOKENS = 2
MAX_MODEL_LEN = 512
NUM_ITERATIONS = 10  # first iterations are the warmup; the last one is the steady state
PATTERN = [1, 2, 3, 4]

from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

llm = LLM(
    model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    dtype="float16",
    tensor_parallel_size=TP,
    max_model_len=MAX_MODEL_LEN,
    max_num_seqs=BATCH_SIZE,
    # Prefix caching would let later iterations reuse the KV of earlier ones and
    # skip the prefill we want to measure.
    enable_prefix_caching=False,
    attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        # The prompt repeats 1,2,3,4, so a 2..4-token suffix always has a match.
        "prompt_lookup_max": len(PATTERN),
        "prompt_lookup_min": 2,
    },
    distributed_executor_backend="external_launcher",  # worker in-process
)

# Exactly INPUT_TOKENS tokens: repeat the pattern and cut to length.
prompt_token_ids = (PATTERN * (INPUT_TOKENS // len(PATTERN) + 1))[:INPUT_TOKENS]
assert len(prompt_token_ids) == INPUT_TOKENS
sampling = SamplingParams(
    max_tokens=OUTPUT_TOKENS, min_tokens=OUTPUT_TOKENS, temperature=0.0, ignore_eos=True
)

engine = llm.llm_engine
step_times_by_iter: list[list[float]] = []

for it in range(NUM_ITERATIONS):
    for b in range(BATCH_SIZE):
        engine.add_request(f"it{it}-req{b}", {"prompt_token_ids": prompt_token_ids}, sampling)

    step_times: list[float] = []
    final_outputs = []
    t_iter = time.perf_counter()
    while engine.has_unfinished_requests():
        t0 = time.perf_counter()
        outs = engine.step()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        n_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        print(
            f"iter {it:2d} step {len(step_times) - 1:2d}: {dt * 1e3:10.2f} ms"
            f"  (output tokens so far: {n_tokens})",
            flush=True,
        )
        final_outputs.extend(o for o in outs if o.finished)
    t_total = time.perf_counter() - t_iter

    print(f"iter {it:2d} total: {t_total * 1e3:10.2f} ms over {len(step_times)} steps\n", flush=True)
    step_times_by_iter.append(step_times)

    assert len(final_outputs) == BATCH_SIZE
    assert all(len(o.prompt_token_ids) == INPUT_TOKENS for o in final_outputs)
    assert all(len(o.outputs[0].token_ids) == OUTPUT_TOKENS for o in final_outputs)

# Per-step summary across iterations: warmup is complete when the later
# iterations show the same times for each step index.
print("Per-step time (ms) by iteration:")
max_steps = max(len(s) for s in step_times_by_iter)
print("iter | " + " | ".join(f"step {i:2d}" for i in range(max_steps)))
for it, steps in enumerate(step_times_by_iter):
    print(f"{it:4d} | " + " | ".join(f"{t * 1e3:7.2f}" for t in steps))

os._exit(0)  # avoids teardown aborts, as in the profiling scripts
