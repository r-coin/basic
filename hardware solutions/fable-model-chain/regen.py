# -*- coding: utf-8 -*-
"""Pass 3 core — the regenerative DS cavity as a transmission resonator.

Below the self-oscillation threshold, a DS cell is a driven resonant cavity:
the current-clamped drain reflects with amplitude r_d = (s+v0)/(s-v0) > 1 (the
DS gain event), the density-clamped source reflects with |r_s| = 1, and one
round trip (lambda_p/2 of propagation) loses a_loss in amplitude. The round-trip
loop gain is

    loop(M) = (s+v0)/(s-v0) * a_loss,   v0 = M s,

and self-oscillation is loop = 1 -> recovers M_th of Eq. (3) exactly. On
resonance the intracavity field of a driven cavity is enhanced by 1/(1-loop),
so the net power gain referenced to the identical passive (M=0) cell is

    G(M) = [ (1 - loop(0)) / (1 - loop(M)) ]^2 .

The few-cycle pulse gain is obtained by recirculating a finite 3-cycle burst
through the same feedback delay line (round-trip delay 2 L/s) and comparing
output energy with and without drift. The nonlinear shallow-water solver
(solver.py) independently confirms this regenerative buildup and locates the
numerically-diffused threshold M_th,num.
"""
import math
import numpy as np
import constants as C
from ds_cell import (plasmon_speed, cell_length, M_threshold,
                     passive_loss_dB_per_half_lambda)


def _cavity(T=C.Tcap):
    s = plasmon_speed()
    tau = C.tau(T)
    L = cell_length(s)
    Mth = M_threshold(L, s, tau)
    a_loss = 10 ** (-passive_loss_dB_per_half_lambda(tau) / 20.0)   # amplitude/round trip
    return s, tau, L, Mth, a_loss


def loop_gain(M, T=C.Tcap):
    s, tau, L, Mth, a_loss = _cavity(T)
    return (1 + M) / (1 - M) * a_loss


def cw_net_gain_dB(M_over_Mth, T=C.Tcap):
    """CW regenerative net power gain referenced to the identical passive cell."""
    s, tau, L, Mth, a_loss = _cavity(T)
    a0 = loop_gain(0.0, T)
    aM = loop_gain(M_over_Mth * Mth, T)
    return 20 * math.log10((1 - a0) / (1 - aM))


def pulse_net_gain_dB(M_over_Mth, T=C.Tcap, ncyc=3, rep_ratio=0.25, dt=0.02):
    """Few-cycle pulse gain via time-domain recirculation in the cavity delay line.
    Round-trip delay = 2 (units L/s); carrier period in those units = 1/f0_n."""
    s, tau, L, Mth, a_loss = _cavity(T)
    f0_n = C.f0 * L / s
    loop = loop_gain(M_over_Mth * Mth, T)
    loop0 = loop_gain(0.0, T)
    rt = 2.0                                  # round-trip delay (L/s units)
    # build a 3-cycle gaussian burst
    span = 6.0 / f0_n
    t = np.arange(0, span, dt)
    width = ncyc / f0_n
    env = np.exp(-0.5 * ((t - span / 2) / (width / 2.355)) ** 2)
    burst = env * np.sin(2 * np.pi * f0_n * (t - span / 2))

    def out_energy(loop):
        # output = sum_k (r_s loop)^k * burst delayed by k*rt; the source reflection
        # r_s = -1 combines with the half-wave round-trip path so copies add in phase.
        N = int((span + 200 * rt) / dt)
        y = np.zeros(N)
        k = 0
        while loop ** k > 1e-4 and k < 4000:
            d = int(k * rt / dt)
            if d >= N:
                break
            seg = ((-loop) ** k) * burst
            m = min(len(seg), N - d)
            y[d:d + m] += seg[:m]
            k += 1
        return np.sum(y ** 2)

    return 10 * math.log10(out_energy(loop) / out_energy(loop0))


def isi_spread_dB(M_over_Mth, f_sym, T=C.Tcap, depth=0.30, nbits=4096, seed=3):
    """Gain-recovery inter-symbol-interference spread over a random bit pattern.

    A '1' pulse depletes the cell's available inversion by `depth`; it recovers
    as exp(-t/tau) between slots. Per-pulse gain tracks the instantaneous loop
    gain (regeneration makes gain steeply sensitive to inversion near threshold),
    so a random pattern produces a spread of per-'1' gains. Returns peak-to-peak
    dB spread of the per-pulse gain over the pattern."""
    s, tau, L, Mth, a_loss = _cavity(T)
    Tslot = 1.0 / f_sym
    rec = math.exp(-Tslot / tau)                 # fraction of depletion surviving a slot
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, nbits)
    M0 = M_over_Mth * Mth
    inv = 1.0                                     # available inversion (1 = full)
    gains = []
    for b in bits:
        # recover toward 1 since previous slot
        inv = 1 - (1 - inv) * rec
        if b:
            # gain set by loop gain scaled by current inversion (depletes drift drive)
            loop = loop_gain(M0 * inv, T)
            loop = min(loop, 0.999)
            g = 20 * math.log10((1 - loop_gain(0.0, T)) / (1 - loop))
            gains.append(g)
            inv = max(inv - depth, 0.0)           # consume inversion
    gains = np.array(gains[len(gains) // 4:])     # drop warm-up
    return gains.max() - gains.min()


if __name__ == "__main__":
    s, tau, L, Mth, a_loss = _cavity()
    print(f"M_th={Mth:.3f}, a_loss(amp/round trip)={a_loss:.3f}, loop(M_th)={loop_gain(Mth):.3f}")
    print("CW net gain vs M/M_th:")
    for r in [0.3, 0.5, 0.7, 0.8]:
        print(f"  {r}: CW {cw_net_gain_dB(r):+.2f} dB | pulse {pulse_net_gain_dB(r):+.2f} dB")
    print("ISI spread (per-pulse, random pattern, 30% depletion):")
    for fs in [0.25e12, 0.5e12, 1.0e12]:
        print(f"  f_sym={fs/1e12:.2f} THz: {isi_spread_dB(0.7, fs):.2f} dB")
