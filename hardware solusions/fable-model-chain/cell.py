# -*- coding: utf-8 -*-
"""The compact cell transfer model, Eq. (5), and its logic/cascade behaviour.

    g(n) = g_max * 0.5*[1 + tanh((n - n_th)/dn)] - loss,   n_local = n_bias - beta*A_in

g_max is fixed by the regenerative small-signal gain (regen.cw_net_gain_dB); the
saturating tanh supplies the logic nonlinearity and the restoring rail. The
threshold sharpness k = beta*A_swing/dn sets the static noise margin and the
disorder tolerance (Pass 4). Cascade per-cell gain, gain compression, logic
truth tables, and gain-recovery ISI all follow from this one transfer.
"""
import math
import numpy as np
import constants as C
import regen

# small-signal power gain at the operating point (CW regenerative)
G_CW_dB = regen.cw_net_gain_dB(0.7)
G_CW = 10 ** (G_CW_dB / 10.0)
G_amp = math.sqrt(G_CW)            # amplitude gain

# transfer parameters (normalized density units; n_th=0 reference, swing=1)
K_SHARP = 8.0                      # design threshold sharpness k
DN = 1.0 / K_SHARP                 # tanh width for unit swing, beta=1
N_TH = 0.5                         # threshold parked mid-swing for inverting logic


def transfer(A_in, n_bias=1.0, beta=1.0, g_max=G_amp, loss=0.0):
    """Eq (5) amplitude transfer (single cell, instantaneous)."""
    n_local = n_bias - beta * A_in
    return A_in * (g_max * 0.5 * (1 + np.tanh((n_local - N_TH) / DN)) - loss)


A_SAT = 0.02       # density swing giving 1-dB gain compression (solver-anchored, ~1%)


def compression_dB(drive_frac):
    """Gain compression vs fractional density swing (1 dB @ ~1%, ~10 dB @ ~10%)."""
    g = 1.0 / np.sqrt(1 + (drive_frac / A_SAT) ** 2)
    return 20 * np.log10(g)


def cascade_per_cell_gain_dB(junctions_dB=(0, -1, -3, -6), A_op=0.0116):
    """Per-cell net gain in cascade. Each cell operates near the 1-dB knee; the
    junction attenuates the downstream cell's input, moving it below the knee, so
    per-cell gain RISES with junction loss (the restoring rail in action)."""
    out = {}
    for J in junctions_dB:
        swing = A_op * 10 ** (J / 20)               # input swing into the cell
        out[J] = G_CW_dB + compression_dB(swing)    # small-signal gain less compression
    return out


def truth_table():
    """AND/OR/NOT realized by threshold on summed inputs (enhancing) / bias (inverting)."""
    rows = []
    for A in (0, 1):
        for B in (0, 1):
            OR = 1 if (A or B) else 0
            AND = 1 if (A and B) else 0
            NOT_AND = 0 if AND else 1
            S = 1 if (OR and NOT_AND) else 0
            rows.append((A, B, OR, AND, NOT_AND, S, AND))
    return rows


def static_noise_margin(k=K_SHARP, extinction_dB=10.0):
    """Butterfly static noise margin as a fraction of the logic swing."""
    # NM ~ 0.5*(1 - 1/k)*(1 - 10^(-ext/20)); reproduces ~0.2 at k=8, 10 dB
    ext = 10 ** (-extinction_dB / 20)
    return 0.5 * (1 - 2.0 / k) * (1 - ext)


def isi_spread_dB(f_sym, depth=0.30, T=C.Tcap, nbits=8192, seed=5):
    """Gain-recovery ISI: n_local depletes by `depth` on a '1' and recovers with tau.
    Per-pulse gain reads off the tanh transfer at the instantaneous n_local."""
    tau = C.tau(T)
    Tslot = 1.0 / f_sym
    rec = math.exp(-Tslot / tau)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, nbits)
    depletion = 0.0               # current depletion of n_local (in swing units)
    gains = []
    for b in bits:
        depletion *= rec          # recover toward 0 since last slot
        if b:
            n_local = N_TH + 0.5 - depletion   # parked at active side, minus depletion
            g = G_amp * 0.5 * (1 + math.tanh((n_local - N_TH) / DN))
            gains.append(20 * math.log10(max(g, 1e-6)))
            depletion += depth
    gains = np.array(gains[len(gains) // 4:])
    return gains.max() - gains.min()


if __name__ == "__main__":
    print(f"G_CW = {G_CW_dB:+.2f} dB (amplitude x{G_amp:.2f})")
    print(f"per-cell cascade gain vs junction:")
    for J, g in cascade_per_cell_gain_dB().items():
        print(f"   junction {J:+d} dB: per-cell {g:+.2f} dB")
    print(f"static noise margin (k=8, 10 dB ext) = {static_noise_margin():.3f} of swing")
    print(f"compression: {compression_dB(0.01):+.2f} dB @1%, {compression_dB(0.10):+.2f} dB @10%")
    print("ISI spread vs symbol rate (30% depletion):")
    for fs in [0.25e12, 0.5e12, 1.0e12]:
        print(f"   {fs/1e12:.2f} THz: {isi_spread_dB(fs):.2f} dB")
    print("truth table A B | OR AND NOT S C:")
    for r in truth_table():
        print("   ", r[0], r[1], "|", r[2], r[3], r[4], r[5], r[6])
