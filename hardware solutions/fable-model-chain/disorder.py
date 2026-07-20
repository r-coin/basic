# -*- coding: utf-8 -*-
"""Pass 4 (disorder) — Monte-Carlo cell yield vs charge-puddle density.

A cell regenerates only if its threshold sharpness clears the floor once the
charge puddle broadens the transition. With design sharpness k0 (=8) at zero
puddle, a puddle of rms density sigma broadens the effective threshold width to
sqrt((n/k0)^2 + sigma^2), so

    k_eff(n, sigma) = n / sqrt((n/k0)^2 + sigma^2),

and the cell yields if k_eff >= k_min (the regeneration floor, ~5). This is the
swing-to-puddle criterion (the yield criterion Appendix A owes), tied directly
to the k >= 8 rule. Yield is the puddle-ensemble fraction that clears it.
"""
import numpy as np
import constants as C

K0 = 8.0          # design sharpness
K_MIN = 5.0       # regeneration floor


def sigma_crit(n):
    """Largest puddle sigma a cell of density n tolerates."""
    return n * np.sqrt(1.0 / K_MIN**2 - 1.0 / K0**2)


def yield_fraction(n_cm2, sigma_lo, sigma_hi, nsamp=20000, seed=11):
    """MC yield: fraction of a log-uniform puddle ensemble [sigma_lo,sigma_hi] that clears."""
    n = n_cm2 * 1e4
    rng = np.random.default_rng(seed)
    logs = rng.uniform(np.log(sigma_lo), np.log(sigma_hi), nsamp)
    sig = np.exp(logs)                       # cm^-2
    sig_m = sig * 1e4                          # m^-2
    return np.mean(sig_m < sigma_crit(n))


if __name__ == "__main__":
    # literature disorder: ultra-flat hBN 2.5e9 .. typical encapsulated 4e10 cm^-2
    lit = (2.5e9, 4e10)
    # broad sweep: down to clean, up to SiO2-substrate-class ~3e11 cm^-2
    broad = (2.5e9, 3e11)
    print("cell yield vs density:")
    for n_cm2 in [1e11, 3e11, 1e12]:
        yl = yield_fraction(n_cm2, *lit)
        yb = yield_fraction(n_cm2, *broad)
        print(f"  n={n_cm2:.0e} cm^-2: literature {100*yl:.0f}% | broad sweep {100*yb:.0f}%  "
              f"(sigma_crit={sigma_crit(n_cm2*1e4)/1e4:.2e} cm^-2)")
