#!/usr/bin/env python3
"""
Quick wall-clock benchmark for SSL regularization methods using float16 tensors.
This is a *synthetic* forward-only benchmark; no gradients are computed.

Methods benchmarked:
- VISReg (Ours): Variance-Invariance-Sketching regularization
- SIGReg: Spectral Integral Gaussian Regularization
- VICReg: Variance-Invariance-Covariance Regularization
- Barlow Twins: Cross-correlation based regularization

Defaults follow the heavy setting discussed:
- V=8 views, B=2048 (adjustable), D=2048, K=2048, m=17 (SIGReg knots)
- float16 tensors on CUDA if available (falls back to CPU).

Examples:
  # Single batch size
  python scripts/bench_cost.py --batch 20000 --views 8 --dim 2048 --k 2048 --knots 17 --warmup 3 --iters 5

  # Sweep batch sizes (comma-separated)
  python scripts/bench_cost.py --batches 2000,5000,10000,20000 --views 8 --dim 2048 --k 2048 --knots 17 --warmup 3 --iters 5
"""

import argparse
import time
import torch
import torch.nn as nn

from visreg import VISReg, SIGReg, SlicedWasserstein as SWD, VICReg, BarlowTwins


def time_module(mod: nn.Module, inp: torch.Tensor, warmup: int, iters: int, device: torch.device):
    mod = mod.to(device)
    inp = inp.to(device)
    mod.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            _ = mod(inp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        times = []
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
                start_event.record()
            _ = mod(inp)
            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))  # ms
        if device.type == "cuda":
            avg_ms = sum(times) / len(times)
            peak_mem = torch.cuda.max_memory_allocated()
        else:
            times = []
            for _ in range(iters):
                t0 = time.time()
                _ = mod(inp)
                times.append((time.time() - t0) * 1000)
            avg_ms = sum(times) / len(times)
            peak_mem = None
    return avg_ms, peak_mem


def time_module_fwd_bwd(
    mod: nn.Module,
    inp: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
):
    mod = mod.to(device)
    mod.eval()
    inp = inp.to(device)
    inp.requires_grad_(True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def _zero_grads():
        mod.zero_grad(set_to_none=True)
        if inp.grad is not None:
            inp.grad = None

    # Warmup (with backward) to stabilize kernels + allocations
    for _ in range(warmup):
        _zero_grads()
        loss = mod(inp)
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    if device.type == "cuda":
        times = []
        for _ in range(iters):
            _zero_grads()
            torch.cuda.synchronize()
            start_event.record()
            loss = mod(inp)
            loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))  # ms
        avg_ms = sum(times) / len(times)
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        times = []
        for _ in range(iters):
            _zero_grads()
            t0 = time.time()
            loss = mod(inp)
            loss.backward()
            times.append((time.time() - t0) * 1000)
        avg_ms = sum(times) / len(times)
        peak_mem = None

    # Important: remove grad flag to avoid surprising the caller if it reuses `inp`
    inp.requires_grad_(False)
    return avg_ms, peak_mem


def _parse_batches(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Benchmark SSL regularization methods (VISReg, SWD, SIGReg, VICReg, Barlow Twins) in float16.")
    ap.add_argument("--batch", type=int, default=None, help="Batch size B (single value).")
    ap.add_argument("--batches", type=str, default=None, help="Comma-separated batch sizes, e.g., 2000,5000,10000.")
    ap.add_argument("--views", type=int, default=8, help="Views V.")
    ap.add_argument("--dim", type=int, default=2048, help="Embedding dim D.")
    ap.add_argument("--k", type=int, default=2048, help="Projections K (for VISReg/SIGReg).")
    ap.add_argument("--knots", type=int, default=17, help="SIGReg knots m.")
    ap.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    ap.add_argument("--iters", type=int, default=5, help="Timed iterations.")
    ap.add_argument("--backward", action="store_true", help="Benchmark forward+backward (autograd) instead of forward-only.")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                    help="Tensor dtype (CUDA only; CPU forces float32).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = ap.parse_args()

    if args.batch is None and args.batches is None:
        args.batch = 2048  # default single batch if nothing provided
    if args.batch is not None and args.batches is not None:
        raise ValueError("Use either --batch or --batches, not both.")

    batch_list = _parse_batches(args.batches) if args.batches is not None else [args.batch]

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    else:
        dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}")
    mode = "fwd+bwd" if args.backward else "fwd"
    print(f"Mode: {mode}")
    print(f"Config: V={args.views}, D={args.dim}, K={args.k}, m={args.knots}, warmup={args.warmup}, iters={args.iters}")
    print()
    ms_hdr = "ms (fwd+bwd)" if args.backward else "ms (fwd)"
    print(f"{'B':>8} | {('VISReg ' + ms_hdr):>14} {'GB':>6} | {('SWD ' + ms_hdr):>14} {'GB':>6} | {('SIGReg ' + ms_hdr):>14} {'GB':>6} | {('VICReg ' + ms_hdr):>14} {'GB':>6} | {('Barlow ' + ms_hdr):>14} {'GB':>6}")
    print("-" * 120)

    # Results storage for all methods
    results = {
        "VISReg": {"B": [], "ms": [], "gb": []},
        "SWD": {"B": [], "ms": [], "gb": []},
        "SIGReg": {"B": [], "ms": [], "gb": []},
        "VICReg": {"B": [], "ms": [], "gb": []},
        "BarlowTwins": {"B": [], "ms": [], "gb": []},
    }

    def fmt(ms):
        return f"{ms:10.2f}" if ms is not None else f"{'OOM':>10}"

    def fmt_mem(b):
        return f"{b/1e9:6.2f}" if b is not None else f"{'-':>6}"

    def run_method(method_name, module_fn, B):
        z = torch.randn(args.views, B, args.dim, device=device, dtype=dtype)
        mod = module_fn()
        if args.backward:
            ms, mem = time_module_fwd_bwd(mod, z, args.warmup, args.iters, device)
        else:
            ms, mem = time_module(mod, z, args.warmup, args.iters, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return ms, mem

    for B in batch_list:
        # Run each method independently
        visreg_ms, visreg_mem = run_method("VISReg", lambda: VISReg(num_projections=args.k), B)
        swd_ms, swd_mem = run_method("SWD", lambda: SWD(num_projections=args.k), B)
        sigreg_ms, sigreg_mem = run_method("SIGReg", lambda: SIGReg(knots=args.knots, num_projections=args.k), B)
        vicreg_ms, vicreg_mem = run_method("VICReg", lambda: VICReg(), B)
        barlow_ms, barlow_mem = run_method("BarlowTwins", lambda: BarlowTwins(), B)

        print(f"{B:8d} | {fmt(visreg_ms)} {fmt_mem(visreg_mem)} | {fmt(swd_ms)} {fmt_mem(swd_mem)} | {fmt(sigreg_ms)} {fmt_mem(sigreg_mem)} | {fmt(vicreg_ms)} {fmt_mem(vicreg_mem)} | {fmt(barlow_ms)} {fmt_mem(barlow_mem)}")

        # Store results
        for name, ms, mem in [("VISReg", visreg_ms, visreg_mem), ("SWD", swd_ms, swd_mem), ("SIGReg", sigreg_ms, sigreg_mem),
                               ("VICReg", vicreg_ms, vicreg_mem), ("BarlowTwins", barlow_ms, barlow_mem)]:
            if ms is not None:
                results[name]["B"].append(B)
                results[name]["ms"].append(ms)
                results[name]["gb"].append(mem / 1e9 if mem is not None else None)

    # Print summary
    print()
    print("=" * 60)
    print("Summary: Max batch size before OOM")
    print("=" * 60)
    for name, data in results.items():
        max_b = max(data["B"]) if data["B"] else 0
        print(f"  {name:12s}: {max_b:,}")

    # Complexity note
    print()
    print("Complexity Analysis:")
    print(f"  VISReg/SWD/SIGReg: O(B × D × K) = O(B × {args.dim} × {args.k})")
    print(f"  VICReg/Barlow: O(B × D²) = O(B × {args.dim}²) = O(B × {args.dim**2:,})")
    if args.k > 0:
        print(f"  → VICReg/Barlow are ~{args.dim / args.k:.2f}× more expensive per sample (leading term)")
    else:
        print("  → VICReg/Barlow are more expensive per sample (leading term), but K was 0.")

if __name__ == "__main__":
    main()
