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

"""CPU tests for CLIP's mixed image+text step workaround (``multimodal/clip.py``)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from spyre_inference.multimodal.clip import has_text_tokens, merge_text_and_vision
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner


def test_text_only_step_runs_text_tower():
    assert has_text_tokens(None, None)
    assert has_text_tokens([], torch.zeros(4, dtype=torch.bool))


def test_image_only_step_skips_text_tower():
    assert not has_text_tokens([torch.ones(2, 8)], torch.ones(2, dtype=torch.bool))


def test_mixed_step_runs_text_tower():
    # The case vLLM 0.28 got wrong: any image made the whole step image-only.
    mask = torch.tensor([True, False, False, False])
    assert has_text_tokens([torch.ones(1, 8)], mask)


def test_merge_keeps_vision_on_image_rows_and_text_elsewhere():
    text = torch.zeros(4, 2)
    vision = torch.ones(4, 2)
    out = merge_text_and_vision(text, vision, torch.tensor([False, True, False, False]))
    assert torch.equal(out, torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))


def test_merge_pads_a_short_mask_with_text_rows():
    out = merge_text_and_vision(torch.zeros(4, 2), torch.ones(4, 2), torch.tensor([True]))
    assert torch.equal(out[0], torch.ones(2))
    assert torch.equal(out[1:], torch.zeros(3, 2))


def test_merge_without_image_rows_returns_text():
    text = torch.randn(3, 2)
    assert merge_text_and_vision(text, torch.ones(3, 2), torch.zeros(3, dtype=torch.bool)) is text


def _runner(features_by_req: dict[str, list], supports_mm_inputs: bool = True):
    return SimpleNamespace(
        supports_mm_inputs=supports_mm_inputs,
        input_batch=SimpleNamespace(req_ids=list(features_by_req)),
        requests={r: SimpleNamespace(mm_features=f) for r, f in features_by_req.items()},
    )


def test_step_with_an_image_request_takes_the_packed_path():
    runner = _runner({"text": [], "image": [object()]})
    assert TorchSpyreModelRunner._step_has_mm_inputs(runner)


def test_text_only_step_keeps_the_rectangle():
    assert not TorchSpyreModelRunner._step_has_mm_inputs(_runner({"a": [], "b": []}))
    assert not TorchSpyreModelRunner._step_has_mm_inputs(
        _runner({"image": [object()]}, supports_mm_inputs=False)
    )
