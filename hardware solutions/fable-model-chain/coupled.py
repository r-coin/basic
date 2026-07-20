# -*- coding: utf-8 -*-
"""Pass 5 (coupled-cell stability) — the last charge-channel parasitic.

Two DS-active segments share one continuous channel with a midpoint density
clamp of strength c in [0,1] (c=0 fully free -> one length-2L cavity; c=1 hard
clamp -> two independent length-L cells). Because the DS increment scales as 1/L
(Eq. 2), the long free cavity has half the per-length drive and therefore twice
the threshold; the effective threshold of the most-unstable coupled mode is

    M_th_eff(c) = M_th_single * (2 - c).

The most-unstable growth rate at bias M is g(c) ~ (s/L)*(M - M_th_eff(c)); it is
negative (stable) for every clamp strength at the working bias and at 0.9 M_th.
"""
import numpy as np
import constants as C
from ds_cell import plasmon_speed, cell_length, M_threshold


def growth_vs_clamp(M_over_Mth, nclamp=41, T=C.Tcap):
    s = plasmon_speed(); tau = C.tau(T); L = cell_length(s)
    Mth = M_threshold(L, s, tau)
    M = M_over_Mth * Mth
    c = np.linspace(0, 1, nclamp)
    Mth_eff = Mth * (2 - c)
    rate = (s / L) * (M - Mth_eff)        # 1/s; <0 stable
    return c, rate, Mth


if __name__ == "__main__":
    for r in [0.7, 0.9]:
        c, rate, Mth = growth_vs_clamp(r)
        print(f"M/M_th={r}: max growth rate over all clamps = {rate.max():.3e} 1/s "
              f"({'STABLE' if rate.max() < 0 else 'UNSTABLE'})")
