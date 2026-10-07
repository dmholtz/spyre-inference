#!/usr/bin/env python3
"""Micro-benchmark: baseline vs Strategy C for conv2d_via_bmm_decomp.

Shape: input (4, 3, 224, 224), weight (768, 3, 32, 32)
  batch=4  => _layouts_supported requires batch==1  => fallback path
  C_in=3   => not stick-aligned (3 % 64 != 0)      => im2col+matmul decomp

Strategies measured
-------------------
A (baseline): F.conv2d compiled via inductor — fires conv2d_via_bmm_decomp as-is.
              spyre::unfold (RT1) + reshape_via_cpu(weight) (RT2) + reshape_via_cpu(bias) (RT3)
              6 boundary crossings.

C (preload):  Mimics SpyreConv2d.process_weights_after_loading: weight and bias are
              reshaped and placed on device ONCE before the timed loop.  Each step
              calls only spyre::unfold (the single inescapable activation round trip)
              then a compiled on-device matmul and bias-add.
              2 boundary crossings.

Usage:
    python scripts/microbench/conv_fallback_bench.py [--warmup N] [--steps M] [--device DEV]
    python scripts/microbench/conv_fallback_bench.py --strategy A   # baseline only
    python scripts/microbench/conv_fallback_bench.py --strategy C   # preload only
"""

import argparse

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

import torch_spyre  # noqa: F401 — registers decompositions, fallback kernels, custom ops

INPUT_SHAPE  = (4, 3, 224, 224)
WEIGHT_SHAPE = (768, 3, 32, 32)
STRIDE       = 32
DTYPE        = torch.float16

N, C_IN, H, W  = INPUT_SHAPE
C_OUT, _, K, _ = WEIGHT_SHAPE
H_OUT = H // STRIDE
W_OUT = W // STRIDE


# ---------------------------------------------------------------------------
# Path confirmation
# ---------------------------------------------------------------------------

def confirm_fallback_path(x: torch.Tensor, weight: torch.Tensor) -> None:
    b, c, c_out = x.shape[0], x.shape[1], weight.shape[0]
    assert not ((b == 1) and (c <= 64) and (c_out % 64 == 0)), (
        "Shape satisfies tiled-layout conditions — this is the happy path, not the fallback."
    )
    print("[path confirmation]")
    print(f"  input  {tuple(x.shape)}  batch={b} != 1  =>  _layouts_supported=False  =>  fallback")
    print(f"  weight {tuple(weight.shape)}  C_in={c} % 64 = {c % 64}  =>  im2col+matmul decomp")
    print()


# ---------------------------------------------------------------------------
# Strategy A — baseline: full F.conv2d through conv2d_via_bmm_decomp
# ---------------------------------------------------------------------------

def make_strategy_a(device):
    def _conv(x, w, b):
        return F.conv2d(x, w, b, stride=STRIDE)
    compiled = torch.compile(_conv, backend="inductor", fullgraph=True, dynamic=False)

    def step(x, weight, bias):
        return compiled(x, weight, bias)

    return step


# ---------------------------------------------------------------------------
# Strategy C — preload: weight and bias reshaped once at "load time",
# each step only pays for spyre::unfold (the inescapable activation RT).
#
# This is what SpyreConv2d.forward_oot now does when _w_2d_dev is set by
# process_weights_after_loading.
# ---------------------------------------------------------------------------

def preload_weight_bias(weight: torch.Tensor, bias: torch.Tensor, device):
    """Mimics SpyreConv2d.process_weights_after_loading for the fallback case.

    Reshape weight and bias on CPU and place them on device once.  At inference
    the only remaining host round trip is the activation (spyre::unfold).
    """
    w_cpu = weight.to("cpu").reshape(C_OUT, C_IN * K * K)
    b_cpu = bias.to("cpu").reshape(1, C_OUT, 1)
    return w_cpu.to(device), b_cpu.to(device)


def make_strategy_c(device, w_2d_dev, bias_3d_dev):
    # Wrap the matmul kernel in torch.compile so it goes through the Spyre
    # inductor backend, matching SpyreConv2d._conv_via_matmul.
    def _matmul_bias(patches, w, b):
        out = torch.matmul(w.unsqueeze(0).expand(N, -1, -1).clone(), patches)
        out = out + b
        return out.reshape(N, C_OUT, H_OUT, W_OUT)

    compiled_mm = torch.compile(_matmul_bias, backend="inductor", fullgraph=True, dynamic=False)

    def step(x, _weight_unused, _bias_unused):
        # Only the activation crosses the host boundary here.
        patches = torch.ops.spyre.unfold(x, (K, K), stride=(STRIDE, STRIDE))
        return compiled_mm(patches, w_2d_dev, bias_3d_dev)

    return step


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def bench(label, step_fn, x, weight, bias, warmup, steps):
    print(f"[{label}] warmup ({warmup} steps) ...")
    for _ in range(warmup):
        step_fn(x, weight, bias)

    print(f"[{label}] profiling ({steps} steps) ...")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        for _ in range(steps):
            with record_function(label):
                step_fn(x, weight, bias)

    print(f"\n[{label}] top ops by self CPU time:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps",  type=int, default=10)
    ap.add_argument("--device", default="spyre")
    ap.add_argument(
        "--strategy", choices=["A", "C", "all"], default="all",
        help="Which strategy to run (default: all)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    weight = torch.randn(WEIGHT_SHAPE, dtype=DTYPE, device=device)
    bias   = torch.randn(C_OUT, dtype=DTYPE, device=device)
    x      = torch.randn(INPUT_SHAPE,  dtype=DTYPE, device=device)

    confirm_fallback_path(x, weight)

    run_all = args.strategy == "all"

    if run_all or args.strategy == "A":
        bench("A-baseline",
              make_strategy_a(device),
              x, weight, bias, args.warmup, args.steps)

    if run_all or args.strategy == "C":
        w_2d_dev, bias_3d_dev = preload_weight_bias(weight, bias, device)
        bench("C-preload",
              make_strategy_c(device, w_2d_dev, bias_3d_dev),
              x, weight, bias, args.warmup, args.steps)


if __name__ == "__main__":
    main()
