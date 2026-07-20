# -*- coding: utf-8 -*-
"""Error rates vs temperature: the quantum-analog unit against the classical
digital fabric, on one axis.

Three machines are compared at every temperature (4-400 K):

  * CLASSICAL DIGITAL (the Part-I half adder): logic levels restored to the
    rail (~10% density swing, ~3.8e3 quanta, qmode) at every cell; decision
    margin = the parent's static noise margin (0.256 of the swing at k=8,
    10 dB extinction); decision noise = the restoring cell's input-referred
    amplifier noise (F = 2 - 1/G at the matched temperature). This BER model
    is Part II's own construction — Part I quotes no bit-error rate — and
    the rail-level restored swing is its stated assumption: nonlinearity is
    free for digital restoration, so restored levels sit ~10x above the
    analog unit's per-input amplitude (3.8e3 vs 4e2 quanta). The quantized
    noise floor makes rail-scale restoration a REQUIREMENT, not a choice: at
    the 38-quantum knee the digital decision would be thermal-noise-limited
    to ~1e-1 BER at 300 K.

  * QUANTUM-ANALOG, 2-BIT DECODE (QMAC-1, qmac): linear interference core,
    N_op quanta per input, three levels on one axis, flash-decoded on-fabric.

  * QUANTUM-ANALOG, 1-BIT DECODE: same hardware, full rail across one
    boundary.

Two more curves bound the design space:
  * the 20-dB-launch-SNR wall: if the launch carries the parent's budgeted
    classical noise (20 dB SNR, modeled as a fixed fraction of the full
    swing), error saturates near 1e-2 at EVERY temperature — classical
    launch noise does not cool away. Quantum operation requires a quiet
    (near-shot-noise) launch; this is the extension's sharpest new hardware
    requirement.
  * 16-slot averaging: the wave-pipelined fabric can spend throughput
    (16 slots at 0.25 THz = 64 ps per MAC, still 1.6e10 MAC/s) by
    accumulating slots in the charge-memory layer. Per-slot noise averages
    as 1/16; the static threshold band does not (qmac.decision_variance).

Temperature structure: T_Q = hbar w0/kB = 48 K is where the noise DENSITY
crosses from thermal- to vacuum-dominated (nbar = 0.58 there); the error
CURVES, being erfc tails, keep falling until nbar << 1/2 and saturate at
their vacuum-set floors below ~15-20 K. That saturation is the quantum limit
made visible, and its observation is bench gate QG2 of the manuscript.
"""
import math
import numpy as np
import qconstants as C
import qmode
import qnoise as Q
import qdecode
import qmac

NM_CLASSICAL = 0.2564          # parent static noise margin (k=8, 10 dB ext)
N_RAIL = qmode.N_of_eps(qmode.EPS_RAIL)     # ~3.8e3 quanta at the rail


def classical_ber(T):
    """Per-decision BER of the rail-restored digital fabric."""
    x_swing = math.sqrt(2 * N_RAIL)
    margin = NM_CLASSICAL * x_swing
    G = qdecode.G_dec(T)
    V = (2 - 1 / G) * (C.nbar(T) + 0.5)
    return Q.tail(margin, V)


def sweep(temps=None, N_op=400):
    """All curves on a common temperature axis; returns dict of arrays."""
    if temps is None:
        temps = np.concatenate([np.arange(4, 60, 2),
                                np.arange(60, 400, 5)]).astype(float)
    out = {"T": np.array(temps, float)}
    out["classical"] = np.array([classical_ber(t) for t in temps])
    out["q2bit"] = np.array([qmac.error_2bit(t, N_op) for t in temps])
    out["q1bit"] = np.array([qmac.error_1bit(t, N_op) for t in temps])
    out["q2bit_launch20"] = np.array(
        [qmac.error_2bit(t, N_op, launch_snr_dB=20) for t in temps])
    out["q2bit_avg16"] = np.array(
        [qmac.error_2bit(t, N_op, n_avg=16) for t in temps])
    out["enob"] = np.array([qmac.analog_enob(t, N_op) for t in temps])
    return out


def threshold_temperature(target, N_op=400, mode="2bit"):
    """Highest temperature at which the decode error meets `target`."""
    f = qmac.error_2bit if mode == "2bit" else qmac.error_1bit
    lo, hi = 4.0, 400.0
    if f(lo, N_op) > target:
        return None
    while hi - lo > 0.5:
        mid = 0.5 * (lo + hi)
        if f(mid, N_op) <= target:
            lo = mid
        else:
            hi = mid
    return lo


def quantum_floor(N_op=400):
    """T->0 error floors (vacuum + device noise only)."""
    return {"2bit": qmac.error_2bit(4.0, N_op),
            "1bit": qmac.error_1bit(4.0, N_op)}


def sensitivity_junction(N_op=400, j_dB=3.0):
    """Errors with junctions at the pessimistic end of the parent's
    -1 to -3 dB band (default assumption elsewhere: -1 dB)."""
    return {f"q2bit_300K_j{j_dB:.0f}dB": qmac.error_2bit(300, N_op, j_dB=j_dB),
            f"q2bit_77K_j{j_dB:.0f}dB": qmac.error_2bit(77, N_op, j_dB=j_dB),
            f"q1bit_300K_j{j_dB:.0f}dB": qmac.error_1bit(300, N_op, j_dB=j_dB),
            f"q1bit_77K_j{j_dB:.0f}dB": qmac.error_1bit(77, N_op, j_dB=j_dB)}


def table(temps=(353, 300, 150, 77, 48, 20, 4), N_op=400):
    rows = []
    for T in temps:
        rows.append({
            "T_K": T,
            "nbar": C.nbar(T),
            "Q": C.Q(T),
            "classical_BER": classical_ber(T),
            "q2bit": qmac.error_2bit(T, N_op),
            "q1bit": qmac.error_1bit(T, N_op),
            "q2bit_avg16": qmac.error_2bit(T, N_op, n_avg=16),
            "enob": qmac.analog_enob(T, N_op),
        })
    return rows


if __name__ == "__main__":
    print(f"rail swing N = {N_RAIL:.0f} quanta; classical margin "
          f"{NM_CLASSICAL} of swing")
    print(f"{'T':>5} {'nbar':>8} {'classical':>11} {'q 2-bit':>11} "
          f"{'q 1-bit':>11} {'2-bit avg16':>12} {'ENOB':>6}")
    for r in table():
        print(f"{r['T_K']:>5.0f} {r['nbar']:>8.3f} {r['classical_BER']:>11.3e} "
              f"{r['q2bit']:>11.3e} {r['q1bit']:>11.3e} "
              f"{r['q2bit_avg16']:>12.3e} {r['enob']:>6.2f}")
    for target in (1e-3, 1e-6, 1e-9):
        t2 = threshold_temperature(target, mode="2bit")
        t1 = threshold_temperature(target, mode="1bit")
        s2 = f"{t2:5.0f} K" if t2 else "  none"
        s1 = f"{t1:5.0f} K" if t1 else "  none"
        print(f"error <= {target:.0e}:  2-bit below {s2} | 1-bit below {s1}")
    fl = quantum_floor()
    print(f"quantum floor (4 K): 2-bit {fl['2bit']:.2e}, 1-bit {fl['1bit']:.2e}")
    print("junction sensitivity (-3 dB vs default -1 dB):")
    for k, v in sensitivity_junction().items():
        print(f"   {k} = {v:.3e}")
