# -*- coding: utf-8 -*-
"""Pass 4 (noise) — phase-insensitive amplifier noise figure, Eq. (7).

A matched-temperature, phase-insensitive linear amplifier of power gain G has the
fundamental noise-figure floor F = 2 - 1/G. The re-sync spacing law follows from
the launch SNR budget: SNR erodes by ~NF per cell, and a station is inserted
before the accumulated erosion eats the margin.
"""
import math
import constants as C
import regen


def noise_figure(G_power):
    """Eq (7): F = 2 - 1/G (linear); returns dB."""
    return 10 * math.log10(2 - 1.0 / G_power)


def resync_spacing(nf_dB, snr_launch_dB=20.0, margin_dB=6.0):
    """Cells between re-sync stations before accumulated noise eats the SNR margin.
    Per-cell added noise is NF; over N restoring cells the noise grows ~NF + 10log10(N)
    (incoherent accumulation past the restoring rail). Allowed erosion = SNR - margin."""
    usable = snr_launch_dB - margin_dB
    return 10 ** ((usable - nf_dB) / 10.0)


if __name__ == "__main__":
    G_dB = regen.cw_net_gain_dB(0.7)
    G = 10 ** (G_dB / 10)
    print(f"operating gain G = {G_dB:+.2f} dB ({G:.2f} x)")
    print(f"noise-figure floor F = {noise_figure(G):.2f} dB  (manuscript 2.8 dB)")
    print(f"total output noise above input floor ~ G*F = {G_dB + noise_figure(G):.1f} dB")
    print("re-sync station spacing vs per-cell NF (SNR 20 dB, 6 dB margin):")
    for nf in [2.8, 3, 6, 10]:
        print(f"   NF={nf:.1f} dB -> spacing ~ {resync_spacing(nf):.0f} cells")
