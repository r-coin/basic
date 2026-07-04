# -*- coding: utf-8 -*-
"""The classical decode block: a 2-bit flash ADC from the parent's own cells.

The decoder quantizes the analog interference output into 1 or 2 bits using
the SAME regenerative threshold cell as every logic gate of the parent fabric
(parent Eq. 5): a comparator is a threshold cell whose n_th is trimmed to a
decision level; its output slams to the rail (digital) on either side.

2-bit ladder (3 levels 0 / N/2 / 2N from the digital QMAC):
    t1 = comparator at the L0|L1 midpoint   (fires on L1 and L2)
    t2 = comparator at the L1|L2 midpoint   (fires on L2 only)
    thermometer (t1,t2) -> binary:  b1 = t2,  b0 = t1 AND NOT t2
so the decoder is two comparators + one inverting cell + one AND cell + one
alignment buffer: FIVE cells — exactly the parent's half-adder block with its
input stage retuned as a flash ladder. In the digital special case the 2-bit
word (b1 b0) IS the binary count A+B: the Part-I half adder re-emerges as the
decoder of a single quantum interference.

Decoder noise contributions (input-referred at the comparator):
  * its own regenerative amplifier noise: (F_dec - 1)(nbar + 1/2),
    F_dec = 2 - 1/G_dec at the working bias (parent Eq. 7);
  * the finite threshold sharpness k: the tanh transition band spans
    ~(full range)/k in x, modeled as a uniform-equivalent variance
    V_k = (range/k)^2 / 12. The parent's logic design rule is k >= 8; the
    flash ladder is a harder analog job, and the design rule here is
    k_dec >= 16 (the parent already quotes k = 16 as achievable).
Static threshold offsets (disorder) are absorbed by the parent's per-cell
top-gate trim against the in-situ reference; residual drift rides on V_k.
"""
import math
import qconstants as C

K_DEC = 16.0                     # decoder comparator sharpness (design rule)


def G_dec(T):
    """Comparator cell power gain at the working bias (linear)."""
    return 10 ** (C.regen_cw_gain_dB(0.7, T) / 10.0)


def F_dec(T):
    """Comparator noise factor, parent Eq. (7)."""
    return 2.0 - 1.0 / G_dec(T)


def decoder_amp_noise(T):
    """Input-referred comparator amplifier noise (per decision/sample)."""
    return (F_dec(T) - 1.0) * (C.nbar(T) + 0.5)


def threshold_band_variance(x_range, k=K_DEC):
    """Static threshold-band variance: the tanh transition band plus residual
    post-trim drift, uniform-equivalent over range/k. STATIC — correlated
    across slots, so it does not average under multi-slot accumulation."""
    return (x_range / k) ** 2 / 12.0


def decoder_input_noise(T, x_range, k=K_DEC):
    """Input-referred decoder noise variance: amplifier + sharpness band."""
    return decoder_amp_noise(T) + threshold_band_variance(x_range, k)


def thermometer_to_binary(t1, t2):
    """(t1,t2) thermometer -> (b1,b0) binary; b0 = t1 AND NOT t2."""
    return t2, int(t1 and not t2)


def decode_level(idx):
    """Level index 0/1/2 -> 2-bit word (b1,b0)."""
    t1 = int(idx >= 1)
    t2 = int(idx >= 2)
    return thermometer_to_binary(t1, t2)


CELL_COUNT = {"comparator": 2, "NOT": 1, "AND": 1, "buffer": 1}


def truth_table():
    """Digital special case: the decoded 2-bit word is the binary count A+B."""
    rows = []
    for A in (0, 1):
        for B in (0, 1):
            level = A + B                    # interference level index
            b1, b0 = decode_level(level)
            rows.append((A, B, level, b1, b0))
    return rows


if __name__ == "__main__":
    T = C.T300
    print(f"comparator gain G_dec(300K)  = {10*math.log10(G_dec(T)):+.2f} dB")
    print(f"comparator NF F_dec(300K)    = {10*math.log10(F_dec(T)):.2f} dB")
    xr = math.sqrt(2 * 2 * 400)   # range for N_op = 400 (level 2N)
    print(f"decoder noise (range {xr:.1f}, k=16): "
          f"V = {decoder_input_noise(T, xr):.2f} "
          f"(amp {(F_dec(T)-1)*(C.nbar(T)+0.5):.2f} + band {(xr/K_DEC)**2/12:.2f})")
    print(f"decoder block: {sum(CELL_COUNT.values())} cells {CELL_COUNT}")
    print("digital truth table  A B | level | b1 b0  (b1b0 = A+B in binary):")
    ok = True
    for A, B, lv, b1, b0 in truth_table():
        ok = ok and (2 * b1 + b0 == A + B)
        print(f"   {A} {B} |   {lv}   |  {b1} {b0}")
    print("decode == A+B for all inputs:", ok)
