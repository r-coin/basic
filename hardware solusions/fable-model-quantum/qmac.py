# -*- coding: utf-8 -*-
"""QMAC-1: the minimal quantum-analog tensor unit — two inputs, one output,
the output decoded to 2 bits by the classical fabric.

Signal chain (all on the parent fabric, all co-drift routed):

  launch A --[input transit]--+
                              |--[combiner w]--[fan-out split]--> t1 comparator
  launch B --[input transit]--+                              --> t2 comparator
                                                                  |
                                              NOT/AND cells <-----+--> (b1, b0)

  * Inputs: comb-locked coherent pulses, N_op quanta at logic 1 (in-fabric,
    after launch coupling — the coupling efficiency itself stays an open item
    of the parent, Section 9.5 there).
  * Interaction: passive coherent interference at a gate-programmed junction
    combiner — amplitude weights sqrt(w), sqrt(1-w) (the analog-computer
    operation). Negative/complex weights are a programmed phase delay on one
    arm (plasmon speed is gate-tunable through parent Eq. 1).
  * Decode: the qdecode flash ladder (2 comparators + NOT + AND + buffer).

THE central architectural finding (computed, not assumed): a regenerative
pre-amplifier before the ladder does NOT pay. Its linear window ends at the
1-dB knee (~38 intracavity quanta, qmode), so amplifying first caps N_op at
~10-20 quanta where thermal noise is ruinous; passively interfering at
N_op ~ 10^2-10^3 and letting the (nonlinear-but-thresholding) comparators do
the gain wins at every temperature. Gain belongs AFTER the decision, exactly
where the parent fabric puts it. Both variants are computed below.

Digital special case (A, B in {0,1} at amplitude weights 1/2, 1/2): the
interference levels are 0, N_op/2, 2 N_op quanta — EQUALLY SPACED in x
(0, sqrt(N_op), 2 sqrt(N_op)) — and the decoded 2-bit word is the binary count
A+B: sum and carry, the Part-I half adder as one interference + one decode.
"""
import math
import qconstants as C
import qmode
import qnoise as Q
import qdecode

# path losses (dB): parent budgets. Input transit: one junction plus a
# quarter-wave of propagation; fan-out: -3 dB split plus one junction. The
# parent budgets junctions at -1 to -3 dB; -1 dB is the design default and
# the -3 dB sensitivity case is reported by qerrors.sensitivity_junction().
JUNCTION_DB = 1.0


def input_transit_dB(T, j_dB=None):
    return (JUNCTION_DB if j_dB is None else j_dB) \
        + C.loss_dB_per_half_lambda(T) / 2.0


def fanout_dB(T, j_dB=None):
    return 3.0 + (JUNCTION_DB if j_dB is None else j_dB) \
        + C.loss_dB_per_half_lambda(T) / 2.0


def preamp_gain(T):
    """Pulse-class regenerative gain (parent: +8 dB streaming at 0.7 M_th).
    Conservative: CW value minus 1.9 dB pulse deficit, as in the parent."""
    return 10 ** ((C.regen_cw_gain_dB(0.7, T) - 1.9) / 10.0)


def compression_gain(eps):
    """Parent cell.py amplitude compression: 1 dB at ~1% intracavity swing."""
    return 1.0 / math.sqrt(1.0 + (eps / 0.02) ** 2)


def _propagate(N_quanta, T, launch_snr_dB=None, N_ref=None, j_dB=None):
    """One input arm up to the combiner input."""
    st = Q.GState(N=N_quanta)
    if launch_snr_dB is not None:
        x_ref = math.sqrt(2 * (N_ref if N_ref else max(N_quanta, 1)))
        st = Q.add_excess(st, Q.launch_excess_variance(x_ref, launch_snr_dB))
    return Q.loss_dB(st, input_transit_dB(T, j_dB), T)


def levels_at_decision(T, N_op=400, w=0.5, preamp=False, launch_snr_dB=None,
                       j_dB=None):
    """Mean x and variance at the comparator input for each digital symbol
    (A,B) in {0,1}^2. Returns dict {(A,B): GState} plus the x range."""
    out = {}
    for A in (0, 1):
        for B in (0, 1):
            sA = _propagate(A * N_op, T, launch_snr_dB, N_op, j_dB)
            sB = _propagate(B * N_op, T, launch_snr_dB, N_op, j_dB)
            st = Q.combine(sA, sB, w)
            if preamp:
                # deterministic compression of the mean (trim absorbs the
                # level shift but not the gap shrinkage), then amp noise.
                # Simplification: added noise is evaluated at the compressed
                # gain; using the small-signal gain instead would be slightly
                # more pessimistic and changes no conclusion (the variant
                # loses by orders of magnitude either way).
                eps = qmode.eps_one() * math.sqrt(max(st.x, 0) ** 2 / 2)
                g2 = (compression_gain(eps)) ** 2
                st = Q.amp(st, preamp_gain(T) * g2, T)
            st = Q.loss_dB(st, fanout_dB(T, j_dB), T)
            out[(A, B)] = st
    x_range = max(s.x for s in out.values())
    return out, x_range


def decision_variance(states, x_range, T, n_avg=1):
    """Common decision variance: worst-case symbol variance + decoder noise.

    n_avg > 1 models multi-slot accumulation in the electronic charge-memory
    layer (parent Section 6.4): each slot is independently sampled through
    the comparator path and the decision is taken once on the accumulated
    mean, so per-slot noise (propagation + sampling amplifier) averages as
    1/n while the STATIC threshold-band/drift term V_k is common to all
    slots and does not average."""
    V_sig = max(s.V for s in states.values())
    V_per_slot = V_sig + qdecode.decoder_amp_noise(T)
    return V_per_slot / n_avg + qdecode.threshold_band_variance(x_range)


def error_2bit(T, N_op=400, preamp=False, launch_snr_dB=None, n_avg=1,
               j_dB=None):
    """Symbol error of the 2-bit decode (digital case, equal input priors)."""
    states, xr = levels_at_decision(T, N_op, 0.5, preamp, launch_snr_dB, j_dB)
    V = decision_variance(states, xr, T, n_avg)
    # collapse the four symbols onto the three levels
    L0 = states[(0, 0)].x
    L1 = 0.5 * (states[(0, 1)].x + states[(1, 0)].x)
    L2 = states[(1, 1)].x
    return Q.symbol_error([L0, L1, L2], [0.25, 0.5, 0.25], V)


def error_1bit(T, N_op=400, preamp=False, launch_snr_dB=None, n_avg=1,
               j_dB=None):
    """1-bit (full-range binary) mode: both inputs driven by the same bit, so
    the whole decode range 0..2*N_op carries a single boundary. (2*N_op = 800
    quanta at the default — a fifth of the fabric rail, 'full-range' names
    the ADC range, not the rail.)"""
    states, xr = levels_at_decision(T, N_op, 0.5, preamp, launch_snr_dB, j_dB)
    V = decision_variance(states, xr, T, n_avg)
    return Q.symbol_error([states[(0, 0)].x, states[(1, 1)].x], [0.5, 0.5], V)


def analog_enob(T, N_op=400, launch_snr_dB=None):
    """Per-shot analog precision of the weighted MAC output: effective number
    of bits over the [0, x_max] range against total rms error (noise only;
    an ideal ADC of matching depth adds quantization below this)."""
    states, xr = levels_at_decision(T, N_op, 0.5, False, launch_snr_dB)
    V = decision_variance(states, xr, T)
    return math.log2(max(xr / math.sqrt(12.0 * V), 1.0))


def weighted_mac_demo(T=77.0, N_op=400, w=0.7):
    """Analog MAC y = sqrt(w) xA + sqrt(1-w) xB on a 5x5 input grid: worst-case
    decoded error of a 2-bit quantization of y, in LSB units."""
    err_max = 0.0
    xr = math.sqrt(2 * N_op) * (math.sqrt(w) + math.sqrt(1 - w))
    eta_in = 10 ** (-input_transit_dB(T) / 10.0)
    eta_fo = 10 ** (-fanout_dB(T) / 10.0)
    scale = math.sqrt(eta_in * eta_fo)
    lsb = scale * xr / 4.0
    for i in range(5):
        for j in range(5):
            aA, aB = i / 4.0, j / 4.0
            sA = _propagate(aA**2 * N_op, T)
            sB = _propagate(aB**2 * N_op, T)
            st = Q.combine(sA, sB, w)
            st = Q.loss_dB(st, fanout_dB(T), T)
            V = st.V + qdecode.decoder_input_noise(T, scale * xr)
            err_max = max(err_max, math.sqrt(V) / lsb)
    return err_max        # rms noise in LSBs of the 2-bit ADC (worst input)


if __name__ == "__main__":
    for T in (353.0, 300.0, 150.0, 77.0, 20.0, 4.0):
        e2 = error_2bit(T)
        e1 = error_1bit(T)
        e2p = error_2bit(T, N_op=19, preamp=True)
        en = analog_enob(T)
        print(f"T={T:5.0f} K: 2-bit {e2:9.3e} | 1-bit {e1:9.3e} | "
              f"preamp-variant (N=19) {e2p:9.3e} | ENOB {en:4.2f}")
    print(f"\n2-bit with 20 dB launch SNR at 77 K: "
          f"{error_2bit(77, launch_snr_dB=20):.3e} (the classical-launch wall)")
    print(f"2-bit, 16-slot averaging at 300 K: {error_2bit(300, n_avg=16):.3e}")
    print(f"weighted MAC (w=0.7) rms noise at 77 K: "
          f"{weighted_mac_demo():.3f} LSB of the 2-bit ADC")
    # digital levels sanity: equally spaced in x
    st, xr = levels_at_decision(300.0)
    xs = sorted(round(s.x, 2) for s in st.values())
    print(f"decision levels x (300 K): {xs} (range {xr:.1f})")
