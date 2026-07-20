# -*- coding: utf-8 -*-
"""Gaussian quadrature calculus for the plasmon signal chain.

Conventions (used by every downstream module):
  * x is the measured quadrature, x = (a + a')/sqrt(2); vacuum variance 1/2.
  * a coherent pulse of N quanta has mean <x> = sqrt(2N) (phase locked to the
    comb clock, which supplies the homodyne reference across the fabric).
  * a thermal state at temperature T has variance nbar(T) + 1/2.

Elementary operations (exact for Gaussian states):
  loss(eta, T):    x -> sqrt(eta) x,  V -> eta V + (1-eta)(nbar+1/2)
  amp(G, T):       x -> sqrt(G)  x,  V -> G V + (G-1)(nbar+1/2)
  combine(w):      x -> sqrt(w) xA + sqrt(1-w) xB,  V -> w VA + (1-w) VB

The amplifier rule is the matched-temperature phase-insensitive amplifier: its
added noise (G-1)(nbar+1/2) referred to the input is (1-1/G)(nbar+1/2), which
reproduces the parent manuscript's noise-figure floor F = 2 - 1/G (Eq. 7
there) at any temperature, and reduces to the Caves quantum limit
(G-1)/2 added quanta as T -> 0. qlindblad.py verifies both elementary rules
against exact Fock-space quantum evolution.

Decision errors: levels are discriminated in x with thresholds trimmed to
midpoints (the parent's in-situ trim); the per-boundary error is the Gaussian
tail  Perr = (1/2) erfc( d / sqrt(2 V) ),  d = half-gap. verify_mc() checks
the assembled symbol error by direct sampling.
"""
import math
import numpy as np
import qconstants as C


class GState:
    """Mean/variance of the x quadrature of one pulse mode."""

    def __init__(self, N=0.0, V=0.5, x=None):
        self.x = math.sqrt(2 * N) if x is None else x
        self.V = V

    def copy(self):
        return GState(x=self.x, V=self.V)


def loss(st, eta, T):
    out = st.copy()
    out.x = math.sqrt(eta) * st.x
    out.V = eta * st.V + (1 - eta) * (C.nbar(T) + 0.5)
    return out


def loss_dB(st, dB, T):
    return loss(st, 10 ** (-abs(dB) / 10.0), T)


def amp(st, G, T):
    out = st.copy()
    out.x = math.sqrt(G) * st.x
    out.V = G * st.V + (G - 1) * (C.nbar(T) + 0.5)
    return out


def combine(stA, stB, w=0.5):
    """Two-port junction combiner, amplitude weights sqrt(w), sqrt(1-w)."""
    out = GState()
    out.x = math.sqrt(w) * stA.x + math.sqrt(1 - w) * stB.x
    out.V = w * stA.V + (1 - w) * stB.V
    return out


def add_excess(st, V_exc):
    out = st.copy()
    out.V = st.V + V_exc
    return out


def launch_excess_variance(x_ref, snr_dB):
    """Classical launch noise: amplitude std = x_ref * 10^(-SNR/20) (SNR is the
    parent's launch power SNR referenced to the full logic amplitude x_ref)."""
    return (x_ref * 10 ** (-snr_dB / 20.0)) ** 2


def tail(d, V):
    """One-sided Gaussian error beyond a threshold at distance d."""
    if V <= 0:
        return 0.0
    return 0.5 * math.erfc(d / math.sqrt(2 * V))


def symbol_error(levels, priors, V):
    """Symbol error for Gaussian levels (means in x) with common variance V and
    midpoint thresholds. levels ascending; priors sum to 1."""
    p = 0.0
    for i, (Lx, pr) in enumerate(zip(levels, priors)):
        e = 0.0
        if i > 0:
            e += tail((Lx - levels[i - 1]) / 2.0, V)
        if i < len(levels) - 1:
            e += tail((levels[i + 1] - Lx) / 2.0, V)
        p += pr * e
    return p


def verify_mc(levels, priors, V, nsamp=2_000_000, seed=7):
    """Monte-Carlo check of symbol_error by direct Gaussian sampling."""
    rng = np.random.default_rng(seed)
    levels = np.asarray(levels, float)
    thresholds = (levels[1:] + levels[:-1]) / 2.0
    sym = rng.choice(len(levels), size=nsamp, p=priors)
    xs = levels[sym] + rng.normal(0, math.sqrt(V), nsamp)
    dec = np.searchsorted(thresholds, xs)
    return float(np.mean(dec != sym))


if __name__ == "__main__":
    T = C.T300
    # a 400-quantum pulse through -3 dB of loss then G=4 gain at 300 K
    st = GState(N=400)
    st = loss_dB(st, 3, T)
    st = amp(st, 4.0, T)
    print(f"chain demo:   x = {st.x:.2f}, V = {st.V:.2f} "
          f"(vacuum 0.5, thermal floor {C.nbar(T)+0.5:.2f})")
    # amplifier noise figure check vs parent Eq. (7)
    G = 10 ** (0.8)
    st_in = GState(N=100, V=C.nbar(T) + 0.5)          # matched-T input floor
    st_out = amp(st_in, G, T)
    snr_in = st_in.x**2 / st_in.V
    snr_out = st_out.x**2 / st_out.V
    F = snr_in / snr_out
    print(f"NF check:     F = {10*math.log10(F):.3f} dB vs 2-1/G = "
          f"{10*math.log10(2-1/G):.3f} dB (parent floor 2.77 dB at G=+9.66dB)")
    # symbol error MC validation
    lv = [0.0, 10.0, 20.0]
    pr = [0.25, 0.5, 0.25]
    V = 4.0
    print(f"symbol error: analytic {symbol_error(lv, pr, V):.5f} "
          f"vs MC {verify_mc(lv, pr, V):.5f}")
