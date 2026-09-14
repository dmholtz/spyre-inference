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

"""Encoder attention warmup coverage: warmup traces every shape serving reaches.

CPU-only, and not about numerics -- ``test_spyre_encoder_attn.py`` covers those,
and covered them while this was broken. Warmup filled every ``(1, L)`` cell
exactly, which satisfies ``_is_b1_fused_sdpa``, so the packed kernels were never
traced at ``B=1`` and compiled mid-request instead.
"""

from types import SimpleNamespace
from typing import cast

import pytest

from spyre_inference.v1.attention.backends import spyre_attn
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    _is_b1_fused_sdpa,
    _ladder_encoder_shape,
)
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    batch_buckets,
    default_encoder_len_buckets,
    encoder_cell_budget,
    next_bucket,
    pick_encoder_attention_shape,
    pooling_warmup_shapes,
)

MAX_NUM_SEQS = 64
MAX_MODEL_LEN = 512
TOKEN_BUDGET = 512


@pytest.fixture(autouse=True)
def _reset_warmup_flag():
    """``mark_warmup_complete`` writes a module global; do not leak it."""
    saved = spyre_attn._warmup_complete
    yield
    spyre_attn._warmup_complete = saved


class TestGateHasTwoSides:
    """``_is_b1_fused_sdpa`` splits B=1 into two compiled shape families."""

    @pytest.mark.parametrize("aligned_len", default_encoder_len_buckets(MAX_MODEL_LEN))
    def test_exact_fill_is_fused_but_one_short_is_not(self, aligned_len):
        # Same (padded_tokens, aligned_len); only real_len differs. Warmup that
        # only ever produces the first line leaves the packed kernels untraced.
        assert _is_b1_fused_sdpa(1, aligned_len, aligned_len, aligned_len)
        assert not _is_b1_fused_sdpa(1, aligned_len, aligned_len, aligned_len - 1)


class TestLadderFallback:
    """An uncovered batch snaps onto the ladder, never onto an invented shape."""

    @pytest.mark.parametrize(
        ("num_seqs", "max_len"),
        [(2, 300), (3, 300), (3, 511), (5, 65), (2, 511), (17, 100)],
    )
    def test_result_is_always_a_ladder_cell(self, num_seqs, max_len):
        batch, length = _ladder_encoder_shape(num_seqs, max_len, MAX_NUM_SEQS, MAX_MODEL_LEN)
        assert batch in batch_buckets(MAX_NUM_SEQS)
        assert length in default_encoder_len_buckets(MAX_MODEL_LEN)
        assert batch >= num_seqs
        assert length >= max_len

    def test_does_not_emit_stick_aligned_non_buckets(self):
        # The regression this replaced: _align_up(300) == 320, which is not a
        # bucket, so every distinct prompt length compiled its own graph.
        assert _ladder_encoder_shape(3, 300, MAX_NUM_SEQS, MAX_MODEL_LEN) == (4, 512)

    def test_silent_during_warmup_and_logs_after(self, caplog):
        """Logged at info: above 8 sequences this is the ordinary path.

        Only batch buckets 1/2/4/8 are warmed at the 512 token budget, so a
        larger batch lands here with nothing wrong and no action to take.
        """
        spyre_attn._warmup_complete = False
        with caplog.at_level("INFO"):
            _ladder_encoder_shape(64, 8, MAX_NUM_SEQS, MAX_MODEL_LEN)
        assert "ladder shape" not in caplog.text, "warmup's own body runs are not news"

        spyre_attn.mark_warmup_complete()
        with caplog.at_level("INFO"):
            _ladder_encoder_shape(3, 300, MAX_NUM_SEQS, MAX_MODEL_LEN)
        assert "ladder shape (B=4, L=512)" in caplog.text
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_log_dedup_key_is_bounded(self, caplog):
        """``info_once`` keys on the args, so they must not carry the request.

        ``num_seqs x max_len`` has thousands of combinations; the ladder has
        ``len(batch_buckets) * len(len_buckets)``. Keying on the request would
        make this an unbounded log and an unbounded ``lru_cache``.
        """
        spyre_attn.mark_warmup_complete()
        with caplog.at_level("INFO"):
            for max_len in range(257, 512):
                _ladder_encoder_shape(3, max_len, MAX_NUM_SEQS, MAX_MODEL_LEN)
        assert caplog.text.count("ladder shape") <= 1


class TestPoolingWarmupCoversBothSides:
    """``_warmup_pooling_bucket_shapes`` traces both sides of the B=1 gate."""

    @staticmethod
    def _run_warmup(shapes, budget=TOKEN_BUDGET):
        """Drive the method against a stub runner.

        Returns one ``(num_tokens, skewed)`` pair per dummy run.
        """
        calls: list[tuple[int, bool]] = []

        def dummy_run(num_tokens, **kwargs):
            assert kwargs.get("force_attention") is True
            calls.append((num_tokens, bool(kwargs.get("create_mixed_batch"))))
            return object(), object()

        runner = SimpleNamespace(
            spyre_shape_bucketer=SimpleNamespace(encoder_shapes=shapes),
            model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
            scheduler_config=SimpleNamespace(
                max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=budget
            ),
            _dummy_run=dummy_run,
            _dummy_pooler_run=lambda hidden: None,
        )
        # Unbound call with a stub self: the method's whole surface is the five
        # attributes above, so this stays host-only instead of building a runner.
        TorchSpyreModelRunner._warmup_pooling_bucket_shapes(cast(TorchSpyreModelRunner, runner))
        assert runner.scheduler_config.max_num_seqs == MAX_NUM_SEQS, "must restore on exit"
        return calls

    def test_every_b1_cell_gets_an_exact_and_a_partial_run(self):
        shapes = pooling_warmup_shapes(
            max_num_seqs=MAX_NUM_SEQS,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=TOKEN_BUDGET,
            len_bucket=default_encoder_len_buckets(MAX_MODEL_LEN),
        )
        b1_cells = [cell for cell in shapes if cell[0] == 1]
        assert b1_cells, "no B=1 cells to check -- the assertion below would be vacuous"
        tokens = [num_tokens for num_tokens, _skewed in self._run_warmup(shapes)]
        for batch_size, prompt_len in shapes:
            if batch_size != 1:
                continue
            assert prompt_len in tokens, f"missing exact fill for (1, {prompt_len})"
            assert prompt_len - 1 in tokens, f"missing partial fill for (1, {prompt_len})"

    def test_no_partial_run_for_batched_cells(self):
        # At B>1 both sides of the gate produce the same packed shapes, so a
        # partial run there would only make warmup slower.
        calls = self._run_warmup([(1, 64), (2, 64), (4, 64)])
        assert calls == [(64, False), (63, False), (128, False), (256, False)]

    def test_over_cell_budget_shape_is_warmed_with_a_skewed_batch(self):
        """``(8, 512)`` busts a 2048 token budget yet is reachable, so warm it skewed.

        One 512-token sequence plus seven single-token ones stays inside the budget
        while carrying the same ``num_seqs`` and ``max_query_len``.
        """
        calls = self._run_warmup([(4, 512), (8, 512)], budget=2048)
        assert calls == [(2048, False), (519, True)]

    def test_shape_too_wide_to_skew_is_skipped_not_mislabelled(self):
        """``create_mixed_batch`` takes ``min(B-1, num_tokens//2)`` decode rows, so a
        cell wider than twice its own length would silently warm a narrower batch.

        ``pooling_warmup_shapes`` cannot emit such a cell (length buckets start at 64),
        so this drives the guard directly.
        """
        assert self._run_warmup([(64, 8)], budget=32) == []

    def test_skewed_fill_is_clamped_to_the_token_budget(self):
        """A budget equal to ``max_model_len`` leaves no room for the skew rows.

        ``(2, 512)`` at a 512 token budget wants 513 tokens, one over ``_dummy_run``'s
        assertion, so the fill is clamped rather than skipped.
        """
        shapes = pooling_warmup_shapes(
            max_num_seqs=MAX_NUM_SEQS,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=MAX_MODEL_LEN,
            len_bucket=default_encoder_len_buckets(MAX_MODEL_LEN),
        )
        assert (2, MAX_MODEL_LEN) in shapes, "the cell this guards is gone; retune the case"
        calls = self._run_warmup(shapes, budget=MAX_MODEL_LEN)
        over = [num_tokens for num_tokens, _skewed in calls if num_tokens > MAX_MODEL_LEN]
        assert not over, f"fills above the budget would trip _dummy_run: {over}"
        assert (MAX_MODEL_LEN, True) in calls, "(2, 512) was skipped, not clamped"

    def test_every_length_bucket_can_lose_a_token(self):
        # The warmup loop subtracts 1 from prompt_len unguarded; a bucket of 1
        # or 0 would make that a zero/negative token count.
        assert min(default_encoder_len_buckets(MAX_MODEL_LEN)) >= 2
        assert min(default_encoder_len_buckets(1)) >= 2


# (max_num_seqs, max_model_len, token budget)
_CONFIGS = [(64, 512, 512), (64, 512, 2048), (64, 512, 8192), (32, 512, 512), (48, 320, 512)]


class TestRoundingTheBatchUpCannotRescueAMiss:
    """``_ladder_encoder_shape``'s exact batch rests on this invariant.

    Warmup is not a single ``B * L`` filter, so assert the property, not a constant.
    """

    @pytest.mark.parametrize(("max_num_seqs", "max_model_len", "budget"), _CONFIGS)
    def test_no_warmed_cell_exceeds_the_cell_budget(self, max_num_seqs, max_model_len, budget):
        cell_budget = encoder_cell_budget(budget)
        for batch, length in pooling_warmup_shapes(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=budget,
            len_bucket=default_encoder_len_buckets(max_model_len),
        ):
            assert batch * length <= cell_budget
            assert batch <= max_num_seqs
            assert length <= max_model_len

    @pytest.mark.parametrize(("max_num_seqs", "max_model_len", "budget"), _CONFIGS)
    def test_every_in_budget_ladder_cell_is_warmed(self, max_num_seqs, max_model_len, budget):
        """The band above the budget is selective; the part below it is not."""
        warmed = set(
            pooling_warmup_shapes(
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=budget,
                len_bucket=default_encoder_len_buckets(max_model_len),
            )
        )
        assert warmed >= {
            (batch, length)
            for batch in batch_buckets(max_num_seqs)
            for length in default_encoder_len_buckets(max_model_len)
            if length <= max_model_len and batch * length <= budget
        }

    @pytest.mark.parametrize(("max_num_seqs", "max_model_len", "budget"), _CONFIGS)
    def test_a_miss_means_the_bucketed_batch_is_not_warmed_either(
        self, max_num_seqs, max_model_len, budget
    ):
        """No reachable batch can be rescued by rounding B onto the ladder."""
        shapes = pooling_warmup_shapes(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=budget,
            len_bucket=default_encoder_len_buckets(max_model_len),
        )
        warmed = set(shapes)
        ladder = default_encoder_len_buckets(max_model_len)
        checked = 0
        for num_seqs in range(1, max_num_seqs + 1):
            for max_len in {1, 13, 64, 65, 300, 400, max_model_len}:
                if max_len > max_model_len:
                    continue
                covered = pick_encoder_attention_shape(
                    num_seqs, max_len, shapes, max_num_seqs, max_model_len, budget
                )
                if covered is not None:
                    continue
                bucketed = (
                    next_bucket(num_seqs, batch_buckets(max_num_seqs)),
                    next_bucket(max_len, ladder),
                )
                assert bucketed not in warmed
                checked += 1
        assert checked, "no misses to check -- the assertion above would be vacuous"
