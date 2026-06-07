"""
findprimes.py — Ryoji Furui's 6n±1 prime sieve (CPU / NumPy reference).

Theory (Ryoji Furui, "The formulation of prime numbers"):
    Every prime > 3 has the form 6n−1 or 6n+1, and every composite of that
    form factors as (6a±1)(6b±1). So it suffices to enumerate the 6n±1
    candidates — one third of the integers — and strike out the composite
    products. This is a 2,3-wheel sieve of Eratosthenes: it never spends work
    marking multiples of 2 or 3.

This is the readable reference implementation. It builds a full mark array of
size 6y+1, so it is meant for small-to-moderate ranges (it needs ~6y bytes of
RAM). For very large ranges or accelerator hardware use primes_mlx.py
(Apple Silicon) or findprimes_gpu.py (CUDA), which sieve in memory-bounded
segments.

CLI:
    python findprimes.py 1000001 2000000        # primes in [6,000,005, 12,000,001]
    python findprimes.py 1000001 2000000 --count
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np


def find_primes(x: int, y: int) -> np.ndarray:
    """Return the primes of the form 6n±1 with n in [x, y].

    Equivalently, every prime in the closed value range [6x−1, 6y+1] except 2
    and 3, returned as a sorted ``int64`` array.

    Parameters
    ----------
    x, y : int
        Inclusive range of ``n`` (``x >= 1`` and ``x <= y``). The smallest
        candidate is ``6x − 1`` and the largest is ``6y + 1``.

    Notes
    -----
    Correctness: a composite candidate ``c`` has a smallest prime factor
    ``p <= sqrt(c) <= sqrt(6y+1)``; striking the multiples of every surviving
    candidate up to ``sqrt(6y+1)`` therefore removes every composite.
    """
    if y < 1 or x < 1 or x > y:
        return np.empty(0, dtype=np.int64)

    hi = 6 * y + 1          # largest value we might emit (6y + 1)
    lo = 6 * x - 1          # smallest value we might emit (6x − 1)

    # The 6n−1 / 6n+1 candidates for n = 1..y, interleaved in ascending order:
    # 5, 7, 11, 13, 17, 19, ...
    k = np.arange(1, y + 1, dtype=np.int64)
    cand = np.empty(2 * y, dtype=np.int64)
    cand[0::2] = 6 * k - 1
    cand[1::2] = 6 * k + 1
    cand = cand[cand <= hi]

    # is_prime[v] is True only for surviving 6n±1 candidates.
    is_prime = np.zeros(hi + 1, dtype=bool)
    is_prime[cand] = True

    # Strike multiples of each surviving candidate up to sqrt(hi). Starting at
    # c*c is the standard Eratosthenes optimisation (smaller multiples already
    # carry a smaller prime factor).
    limit = math.isqrt(hi)
    for c in cand:
        if c > limit:
            break
        if is_prime[c]:
            is_prime[c * c :: c] = False

    return cand[(cand >= lo) & is_prime[cand]]


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Ryoji Furui's 6n±1 prime sieve (NumPy reference)."
    )
    ap.add_argument("x", nargs="?", type=int,
                    help="lower n; smallest candidate is 6x−1")
    ap.add_argument("y", nargs="?", type=int,
                    help="upper n; largest candidate is 6y+1")
    ap.add_argument("--count", action="store_true",
                    help="print only the number of primes found")
    args = ap.parse_args()

    x = args.x if args.x is not None else int(input("Find prime numbers from 6n-1, n= "))
    y = args.y if args.y is not None else int(input("Find prime numbers to 6n+1, n= "))

    start = time.time()
    primes = find_primes(x, y)
    elapsed = time.time() - start

    if args.count:
        print(len(primes))
    else:
        print(primes)
    print(f"{len(primes):,} primes in [{6 * x - 1:,}, {6 * y + 1:,}]  ({elapsed:.4f}s)")


if __name__ == "__main__":
    _main()
