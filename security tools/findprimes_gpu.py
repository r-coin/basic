"""
findprimes_gpu.py — Ryoji Furui's 6n±1 prime sieve on an NVIDIA GPU (CuPy).

Same algorithm and return contract as findprimes.find_primes, but the mark
array lives in GPU memory and the strided strike-outs run as CuPy kernels.

Theory (Ryoji Furui, "The formulation of prime numbers"):
    Every prime > 3 is 6n−1 or 6n+1; every composite of that form factors as
    (6a±1)(6b±1). Enumerate the 6n±1 candidates and strike the composites.

Requirements:
    pip install cupy-cuda12x      # (or the wheel matching your CUDA toolkit)
    A CUDA-capable NVIDIA GPU with enough memory to hold a (6y+1)-byte array.
    For ranges too large to fit, use the segmented primes_mlx.py design or
    add a segmented loop here.

Notes:
    The previous version mixed a numba.cuda kernel with batched host arrays in a
    way that indexed the wrong elements; this version uses CuPy's strided
    assignment, which is both correct and the idiomatic way to express a sieve
    on the GPU. We deliberately strike multiples of every small candidate
    (not only primes): the extra writes are cheap and idempotent, and skipping
    the per-candidate primality test avoids a host/device sync on each step.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:  # pragma: no cover - depends on local CUDA install
    HAS_CUPY = False


def find_primes_gpu(x: int, y: int) -> np.ndarray:
    """Return primes of the form 6n±1 with n in [x, y] as a NumPy int64 array.

    Equivalent to findprimes.find_primes, computed on the GPU via CuPy.
    """
    if not HAS_CUPY:
        raise RuntimeError(
            "CuPy is not available. Install the wheel that matches your CUDA "
            "toolkit, e.g. `pip install cupy-cuda12x`, and ensure an NVIDIA GPU "
            "is visible."
        )
    if y < 1 or x < 1 or x > y:
        return np.empty(0, dtype=np.int64)

    hi = 6 * y + 1
    lo = 6 * x - 1

    # Build the 6n−1 / 6n+1 candidates directly on the device.
    k = cp.arange(1, y + 1, dtype=cp.int64)
    cand = cp.empty(2 * y, dtype=cp.int64)
    cand[0::2] = 6 * k - 1
    cand[1::2] = 6 * k + 1
    cand = cand[cand <= hi]

    is_prime = cp.zeros(hi + 1, dtype=cp.bool_)
    is_prime[cand] = True

    # Drive the (short) outer loop from the host using just the small factor
    # candidates up to sqrt(hi); each strike-out is a single strided GPU kernel.
    limit = math.isqrt(hi)
    small = cp.asnumpy(cand[cand <= limit])
    for c in small.tolist():
        is_prime[c * c :: c] = False

    primes = cand[is_prime[cand] & (cand >= lo)]
    return cp.asnumpy(primes).astype(np.int64)


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Ryoji Furui's 6n±1 prime sieve on an NVIDIA GPU (CuPy)."
    )
    ap.add_argument("x", nargs="?", type=int,
                    help="lower n; smallest candidate is 6x−1")
    ap.add_argument("y", nargs="?", type=int,
                    help="upper n; largest candidate is 6y+1")
    ap.add_argument("--count", action="store_true",
                    help="print only the number of primes found")
    args = ap.parse_args()

    if not HAS_CUPY:
        print("CuPy/CUDA not available. Install cupy-cudaXX on a machine with an "
              "NVIDIA GPU, or use findprimes.py (CPU) / primes_mlx.py (Apple).")
        return

    x = args.x if args.x is not None else int(input("Find prime numbers from 6n-1, n= "))
    y = args.y if args.y is not None else int(input("Find prime numbers to 6n+1, n= "))

    start = time.time()
    primes = find_primes_gpu(x, y)
    elapsed = time.time() - start

    if args.count:
        print(len(primes))
    else:
        print(primes)
    print(f"{len(primes):,} primes in [{6 * x - 1:,}, {6 * y + 1:,}]  ({elapsed:.4f}s)")


if __name__ == "__main__":
    _main()
