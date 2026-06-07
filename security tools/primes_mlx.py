"""
primes_mlx.py — Ryoji Furui's 6n±1 prime sieve, MLX-accelerated for Apple Silicon.

Single file. Auto-tunes for any Apple Silicon chip (M1, M2, M3, M4, M5+) and
any memory configuration (8 GB through 192 GB Ultra). No deps beyond
`mlx` and `numpy`.

Theory (Ryoji Furui, "The formulation of prime numbers"):
    Every prime > 3 is of the form 6n−1 or 6n+1.
    Every composite of that form factors as (6a±1)(6b±1).
    Sieve = candidates − {products}.

Strategy:
    1. Detect chip family + RAM via sysctl/system_profiler.
    2. Pick a memory budget (~60% of RAM, 8 MB floor).
    3. If the full 0..6y+1 mark array fits the budget, run FULL mode
       (one sieve, scatter all factors into it).
    4. Otherwise, run SEGMENTED mode: split [0, 6y+1] into blocks sized
       to fit the budget and sieve each block in turn. Memory footprint
       becomes O(segment_size + factors), independent of y.

Revision notes:
    * Trial factors are now restricted to the PRIMES ≤ √(6y+1) rather than to
      every 6n±1 candidate ≤ √(6y+1). Every composite 6n±1 has a prime factor
      in that range, so the result is identical while the GPU does strictly
      fewer scatter writes (about a 3-4× reduction in factor count).
"""

from __future__ import annotations

import math
import platform
import subprocess
import time
from typing import Optional

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


# ===========================================================================
# Hardware detection
# ===========================================================================
def detect_hardware() -> dict:
    """Probe Apple Silicon chip + RAM + GPU cores. Falls back to safe defaults."""
    info = {
        "chip": "unknown",
        "chip_class": "M1",
        "is_apple_silicon": False,
        "mem_gb": 16.0,
        "gpu_cores": None,
    }
    if platform.system() != "Darwin":
        return info

    try:
        info["chip"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, timeout=2,
        ).strip()
        chip_l = info["chip"].lower() if info["chip"] else ""
        info["is_apple_silicon"] = "apple" in chip_l
    except Exception:
        pass

    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, timeout=2,
        ).strip())
        info["mem_gb"] = mem_bytes / (1024 ** 3)
    except Exception:
        pass

    # GPU core count (best effort).
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True, timeout=4,
        )
        for line in out.splitlines():
            if "Total Number of Cores" in line:
                try:
                    info["gpu_cores"] = int(line.split(":")[-1].strip())
                except Exception:
                    info["gpu_cores"] = None
                break
    except Exception:
        pass

    chip_l = info["chip"].lower() if info["chip"] else ""
    if "m5" in chip_l:
        info["chip_class"] = "M5"
    elif "m4" in chip_l:
        info["chip_class"] = "M4"
    elif "m3" in chip_l:
        info["chip_class"] = "M3"
    elif "m2" in chip_l:
        info["chip_class"] = "M2"
    else:
        info["chip_class"] = "M1"
    return info


# ===========================================================================
# Auto-tuning
# ===========================================================================
def autotune(max_val: int, hw: dict,
             memory_fraction: float = 0.60) -> dict:
    """Pick (mode, segment_size, factor_batch, dtype) for max_val on this hardware.

    memory_fraction is the share of RAM we're willing to dedicate to the sieve
    working set. 0.60 leaves room for OS, Python, MLX scratch, and other apps.
    """
    mem_gb = hw["mem_gb"]
    chip_class = hw["chip_class"]

    # Sane minimum budget so a misdetected RAM size doesn't strand us with
    # zero working memory.
    min_budget = 8 * 1024 ** 2
    budget_bytes = max(int(mem_gb * (1024 ** 3) * memory_fraction), min_budget)

    # Sieve costs 1 byte per value (uint8). Candidate arrays cost ~4 bytes per
    # candidate (int32) and there are ~max_val/3 of them.
    sieve_full_bytes = int(max_val) + 1
    candidate_bytes = (max_val // 3 + 1) * 4
    full_total = sieve_full_bytes + 2 * candidate_bytes  # plus scratch headroom

    if full_total < budget_bytes:
        # Plenty of room: full mode.
        mode = "full"
        segment_size = sieve_full_bytes
    else:
        # Segmented mode. Sieve fits in budget/8 to leave room for candidate
        # arrays and GPU scratch buffers; minimum 4 MB to amortise launches.
        seg = max(1 << 22, budget_bytes // 8)
        seg = min(seg, sieve_full_bytes)
        # Round to multiple of 6 so the 6n±1 stride aligns segment boundaries.
        seg = ((seg + 5) // 6) * 6
        segment_size = seg
        mode = "segmented"

    # Per-batch factor count (kernel launch granularity). Larger chips have
    # better launch overhead and wider GPUs, so we feed them more per batch.
    base = {"M1": 64, "M2": 128, "M3": 256, "M4": 512, "M5": 1024}.get(chip_class, 128)
    if mem_gb >= 64:
        factor_batch = base * 4
    elif mem_gb >= 32:
        factor_batch = base * 2
    elif mem_gb >= 16:
        factor_batch = base
    else:
        factor_batch = max(32, base // 2)

    # int32 if max_val fits, else int64. int32 halves memory traffic.
    use_int32 = max_val < (1 << 31)

    return {
        "mode": mode,
        "segment_size": segment_size,
        "factor_batch": factor_batch,
        "use_int32": use_int32,
        "budget_bytes": budget_bytes,
    }


# ===========================================================================
# Helpers
# ===========================================================================
def _candidates_np(n_lo: int, n_hi: int, dtype=None) -> np.ndarray:
    """Sorted 6n−1 ∪ 6n+1 for n in [n_lo, n_hi]."""
    if dtype is None:
        dtype = np.int64
    if n_hi < n_lo:
        return np.empty(0, dtype=dtype)
    n_arr = np.arange(n_lo, n_hi + 1, dtype=dtype)
    out = np.empty(2 * n_arr.size, dtype=dtype)
    out[0::2] = 6 * n_arr - 1
    out[1::2] = 6 * n_arr + 1
    return out


def _prime_factors_upto(limit: int, dtype=None) -> np.ndarray:
    """Primes p with 3 < p ≤ limit — the only trial factors the sieve needs.

    Every composite of the form 6n±1 has a prime factor ≤ √(6y+1), so striking
    the multiples of just these primes removes every composite. 2 and 3 never
    divide a 6n±1 value, so they are excluded. The factor sieve itself is a
    cheap CPU Eratosthenes over [0, limit] (limit = √(6y+1) is tiny relative
    to the main range).
    """
    if dtype is None:
        dtype = np.int64
    if limit < 5:
        return np.empty(0, dtype=dtype)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, math.isqrt(limit) + 1):
        if sieve[i]:
            sieve[i * i :: i] = False
    primes = np.flatnonzero(sieve)
    primes = primes[primes > 3]
    return primes.astype(dtype)


def _np_dtype_for(use_int32: bool):
    return np.int32 if use_int32 else np.int64


def _mx_dtype_for(use_int32: bool):
    return mx.int32 if use_int32 else mx.int64


# ===========================================================================
# Full mode — entire sieve fits in unified memory
# ===========================================================================
def _find_primes_full(x: int, y: int, plan: dict) -> np.ndarray:
    max_val = 6 * y + 1
    sqrt_max = int(math.isqrt(max_val))
    np_dt = _np_dtype_for(plan["use_int32"])
    mx_dt = _mx_dtype_for(plan["use_int32"])

    is_composite = mx.zeros(max_val + 1, dtype=mx.uint8)

    # Only the primes ≤ √max are needed as trial factors.
    factors_host = _prime_factors_upto(sqrt_max, np_dt)

    full_cands_host = _candidates_np(1, y, dtype=np_dt)
    full_cands = mx.array(full_cands_host).astype(mx_dt)

    factor_batch = plan["factor_batch"]
    for fi in range(0, factors_host.size, factor_batch):
        batch = factors_host[fi : fi + factor_batch]
        for f in batch:
            f = int(f)
            if f * f > max_val:
                break
            max_b = max_val // f
            lo = int(np.searchsorted(full_cands_host, f, side="left"))
            hi = int(np.searchsorted(full_cands_host, max_b, side="right"))
            if hi <= lo:
                continue
            b_vals = full_cands[lo:hi]
            # By construction f * b ≤ f * max_b ≤ max_val, so no clipping needed.
            products = mx.array(f, dtype=mx_dt) * b_vals
            ones = mx.ones(products.size, dtype=mx.uint8)
            is_composite[products] = ones
        mx.eval(is_composite)  # bound graph depth

    targets_host = _candidates_np(x, y, dtype=np_dt)
    targets = mx.array(targets_host).astype(mx_dt)
    target_marks = is_composite[targets]
    mx.eval(target_marks)
    return targets_host[np.asarray(target_marks) == 0].astype(np.int64)


# ===========================================================================
# Segmented mode — sieve is too large for memory, process in blocks
# ===========================================================================
def _find_primes_segmented(x: int, y: int, plan: dict) -> np.ndarray:
    max_val = 6 * y + 1
    target_lo = 6 * x - 1
    target_hi = 6 * y + 1
    sqrt_max = int(math.isqrt(max_val))
    np_dt = _np_dtype_for(plan["use_int32"])
    mx_dt = _mx_dtype_for(plan["use_int32"])

    # Only the primes ≤ √max are needed as trial factors.
    factors_host = _prime_factors_upto(sqrt_max, np_dt)

    segment_size = plan["segment_size"]
    n_segments = (max_val + 1 + segment_size - 1) // segment_size
    chunks: list[np.ndarray] = []

    for seg_idx in range(n_segments):
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, max_val + 1)
        seg_len = seg_end - seg_start
        if seg_len <= 0:
            continue

        # Skip segments outside the target range.
        if seg_end - 1 < target_lo or seg_start > target_hi:
            continue

        sieve = mx.zeros(seg_len, dtype=mx.uint8)

        for f in factors_host:
            f = int(f)
            if f * f > max_val:
                break
            # b range for products in [seg_start, seg_end): b in [b_lo, b_hi].
            b_lo = max(f, (seg_start + f - 1) // f)
            b_hi = (seg_end - 1) // f
            if b_hi < b_lo:
                continue

            # b ∈ {6k−1} ∩ [b_lo, b_hi]. By construction every product
            # f * (6k−1) lies in [seg_start, seg_end), so offsets ∈ [0, seg_len).
            k_lo_m = (b_lo + 6) // 6
            k_hi_m = (b_hi + 1) // 6
            if k_hi_m >= k_lo_m and k_lo_m >= 1:
                k = mx.arange(k_lo_m, k_hi_m + 1, dtype=mx_dt)
                cand_m = 6 * k - 1
                products = mx.array(f, dtype=mx_dt) * cand_m
                offsets = products - mx.array(seg_start, dtype=mx_dt)
                ones = mx.ones(offsets.size, dtype=mx.uint8)
                sieve[offsets] = ones

            # b ∈ {6k+1} ∩ [b_lo, b_hi].
            k_lo_p = (b_lo + 4) // 6
            k_hi_p = (b_hi - 1) // 6
            if k_hi_p >= k_lo_p and k_lo_p >= 1:
                k = mx.arange(k_lo_p, k_hi_p + 1, dtype=mx_dt)
                cand_p = 6 * k + 1
                products = mx.array(f, dtype=mx_dt) * cand_p
                offsets = products - mx.array(seg_start, dtype=mx_dt)
                ones = mx.ones(offsets.size, dtype=mx.uint8)
                sieve[offsets] = ones

        mx.eval(sieve)
        sieve_host = np.asarray(sieve)

        # Extract primes from the intersection of this segment with target range.
        # Wider n window than strictly needed, then filter exactly — avoids
        # missing 6n+1 at the lower edge or 6n−1 at the upper edge.
        block_lo = max(seg_start, target_lo)
        block_hi = min(seg_end - 1, target_hi)
        n_lo = max(x, (block_lo + 4) // 6)
        n_hi = min(y, (block_hi + 1) // 6)
        if n_hi < n_lo:
            continue

        targets_block = _candidates_np(n_lo, n_hi, dtype=np.int64)
        targets_block = targets_block[
            (targets_block >= block_lo) & (targets_block <= block_hi)
        ]
        if targets_block.size == 0:
            continue
        offsets = (targets_block - seg_start).astype(np.int64)
        primes_block = targets_block[sieve_host[offsets] == 0]
        if primes_block.size > 0:
            chunks.append(primes_block)

        # Encourage MLX to free the segment buffer before allocating the next.
        del sieve, sieve_host

    if chunks:
        return np.concatenate(chunks).astype(np.int64)
    return np.empty(0, dtype=np.int64)


# ===========================================================================
# Top-level API
# ===========================================================================
def find_primes_mlx(x: int, y: int,
                    verbose: bool = False,
                    memory_fraction: float = 0.60,
                    factor_batch: Optional[int] = None,
                    segment_size: Optional[int] = None) -> np.ndarray:
    """Return primes in [6x−1, 6y+1] (excluding 2, 3) using MLX, auto-tuned.

    Parameters
    ----------
    x, y : int
        Range of n. Smallest candidate is 6x−1, largest is 6y+1.
    verbose : bool
        Print the chosen plan (chip, mode, segment size, factor batch).
    memory_fraction : float
        Share of physical RAM to use for the sieve working set (default 0.60).
    factor_batch, segment_size : int, optional
        Manual overrides of the auto-tuned values.
    """
    if not HAS_MLX:
        raise RuntimeError(
            "mlx not installed. Install with: pip install mlx (Apple Silicon only)."
        )
    if y < 1 or x < 1 or x > y:
        return np.empty(0, dtype=np.int64)

    max_val = 6 * y + 1
    hw = detect_hardware()
    plan = autotune(max_val, hw, memory_fraction=memory_fraction)
    if factor_batch is not None:
        plan["factor_batch"] = factor_batch
    if segment_size is not None:
        # User override forces segmented mode if smaller than the full sieve.
        seg = ((segment_size + 5) // 6) * 6
        plan["segment_size"] = min(seg, max_val + 1)
        plan["mode"] = "segmented" if plan["segment_size"] < max_val + 1 else "full"

    if verbose:
        print(f"[primes_mlx] {hw['chip']!r}  class={hw['chip_class']}  "
              f"RAM={hw['mem_gb']:.1f} GB  GPU cores={hw['gpu_cores']}")
        sieve_b = plan["segment_size"] if plan["mode"] == "segmented" else (max_val + 1)
        print(f"[primes_mlx] mode={plan['mode']}  "
              f"sieve buffer={sieve_b/1024**2:.1f} MB  "
              f"budget={plan['budget_bytes']/1024**3:.2f} GB  "
              f"factor_batch={plan['factor_batch']}  "
              f"int32={plan['use_int32']}")

    if plan["mode"] == "full":
        return _find_primes_full(x, y, plan)
    return _find_primes_segmented(x, y, plan)


def warmup() -> None:
    """Compile/cache MLX kernels by running a tiny problem first."""
    if HAS_MLX:
        find_primes_mlx(1, 100)


# ===========================================================================
# CLI
# ===========================================================================
def _main() -> None:
    if not HAS_MLX:
        print("MLX not available. Install with: pip install mlx (Apple Silicon only)")
        return
    print("Detecting hardware and warming up MLX kernels...")
    hw = detect_hardware()
    print(f"  Chip:  {hw['chip']}  ({hw['chip_class']})")
    print(f"  RAM:   {hw['mem_gb']:.1f} GB")
    if hw["gpu_cores"]:
        print(f"  GPU:   {hw['gpu_cores']} cores")
    warmup()

    x = int(input("Find prime numbers from 6n-1, n= "))
    y = int(input("Find prime numbers to 6n+1, n= "))
    start = time.time()
    primes = find_primes_mlx(x, y, verbose=True)
    elapsed = time.time() - start
    print(primes)
    print(f"{len(primes)} primes")
    print(f"{elapsed:.4f} s")


if __name__ == "__main__":
    _main()
