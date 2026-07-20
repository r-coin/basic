# -*- coding: utf-8 -*-
"""Operating point and quantum scales for the Fable Computer quantum extension.

Standalone companion to fable-model-chain/constants.py: the classical operating
point is reproduced here verbatim (same values, same conventions) and the
quantum scales of the carrier are added. All SI unless noted; densities in m^-2.

New physics owned by this module:
  * hbar*w0 and the thermal crossover temperature T_Q = hbar*w0/kB ~ 48 K
  * thermal occupation nbar(T) of the 1-THz mode (Bose-Einstein)
  * tau_q(T): momentum-relaxation lifetime with the impurity-scattering
    saturation below T_sat = 150 K that the parent manuscript flags as the
    cold-side limit of the 1/T phonon scaling (Section 8.3 there). The parent
    chain's tau(T) had no saturation because it never operated below 200 K.
  * per-gate propagation loss and the regenerative CW gain re-derived vs T
    (self-checked against fable-model-chain values at 300/353 K).
"""
import math

# --- fundamental constants (CODATA) ---
e = 1.602176634e-19        # C
hbar = 1.054571817e-34     # J s
kB = 1.380649e-23          # J/K
eps0 = 8.8541878128e-12    # F/m

# --- graphene / stack (identical to the parent chain) ---
vF = 1.0e6                 # m/s
eps_z = 3.32               # design value, top of the reported 3.3-3.8 band
d = 10e-9                  # m, gate-graphene hBN spacer

# --- operating point (identical to the parent chain) ---
n_op = 1.0e12 * 1e4        # m^-2
f0 = 1.0e12                # Hz
w0 = 2 * math.pi * f0      # rad/s
T300 = 300.0               # K
Tcap = 353.0               # K (80 C)
tau_300 = 1.0e-12          # s (best-case material, as in the parent)
f_sym = 0.25e12            # Hz, baseline symbol rate
slot = 1.0 / f_sym         # 4 ps
cell_area = 1e-8 * 1e-4    # m^2 per cell (1e-8 cm^2: ~576 nm x ~1.7 um)

# --- quantum scales of the carrier ---
E_quantum = hbar * w0            # J, one plasmon at 1 THz (~4.14 meV)
T_Q = E_quantum / kB             # K, thermal crossover ~ 48 K

# --- cold-side lifetime model ---
T_sat = 150.0                    # K, impurity-scattering saturation (parent Sec. 8.3)


def nbar(T):
    """Bose-Einstein occupation of the carrier mode at temperature T."""
    if T <= 0:
        return 0.0
    return 1.0 / math.expm1(E_quantum / (kB * T))


def tau_q(T):
    """Momentum-relaxation lifetime: phonon-limited 1/T above T_sat, flat below.

    Above T_sat this reproduces the parent chain's tau(T) exactly; below it the
    lifetime saturates at tau(T_sat) = 2 ps (impurity floor), so Q saturates at
    ~12.6. Premium (graphite-gated) material could do better; that option is
    carried as an upside, not assumed.
    """
    return tau_300 * (T300 / max(T, T_sat))


def Q(T):
    """Plasmon quality factor w0 * tau_q(T)."""
    return w0 * tau_q(T)


def E_F(n=n_op):
    kF = math.sqrt(math.pi * n)
    return hbar * vF * kF


def m_star(n=n_op):
    """Effective mass E_F/vF^2 for massless carriers."""
    return E_F(n) / vF**2


def plasmon_speed(n=n_op):
    """Gated local-RPA acoustic-plasmon speed, parent Eq. (1)."""
    kF = math.sqrt(math.pi * n)
    g = 2 * e**2 * kF * d / (math.pi * eps0 * eps_z * hbar * vF)
    return vF * (1 + g) / math.sqrt(1 + 2 * g)


s_design = plasmon_speed()       # ~2.33e6 m/s
lambda_p = s_design / f0         # ~2.33 um


def loss_dB_per_half_lambda(T):
    """Per-gate propagation loss: one lambda_p/2 of travel — 13.64/Q dB
    (the parent's 27.3/Q figure is per FULL lambda_p; per half: 2.17 dB at
    300 K, 2.56 at 353 K, saturating at 1.09 dB below 150 K with tau_q)."""
    # power decays exp(-t/tau); travel time for lambda_p/2 is 1/(2 f0), so
    # loss = 10*log10(e) * (1/(2 f0 tau)) = 10*log10(e)*pi/Q = 13.64/Q dB...
    # (parent quotes 27.29/Q per full lambda_p, i.e. 2.17 dB per half at 300 K)
    return 10 * math.log10(math.e) * math.pi / Q(T)


def M_threshold(T):
    """DS self-oscillation threshold, parent Eq. (3): M_th ~ 1/(8 f0 tau)."""
    return 1.0 / (8 * f0 * tau_q(T))


def regen_cw_gain_dB(M_over_Mth, T):
    """CW regenerative net power gain vs the identical passive cell (parent
    regen.py, Fabry-Perot below threshold). Self-checks to +9.66 dB at
    (0.7, 353 K)."""
    a_loss = 10 ** (-loss_dB_per_half_lambda(T) / 20.0)
    M = M_over_Mth * M_threshold(T)
    loop0 = a_loss
    loopM = (1 + M) / (1 - M) * a_loss
    return 20 * math.log10((1 - loop0) / (1 - loopM))


if __name__ == "__main__":
    print(f"hbar*w0        = {E_quantum/e*1e3:.2f} meV = {E_quantum:.3e} J")
    print(f"T_Q            = {T_Q:.1f} K   (thermal crossover)")
    for T in (4, 20, 48, 77, 150, 300, 353):
        print(f"  nbar({T:>3} K) = {nbar(T):10.4g}   Q = {Q(T):5.2f}   "
              f"loss/half-lambda = {loss_dB_per_half_lambda(T):4.2f} dB")
    print(f"s_design       = {s_design:.3e} m/s (parent 2.331e6)")
    print(f"E_F            = {E_F()/e*1e3:.1f} meV (parent 116.7)")
    print(f"M_th(353K)     = {M_threshold(Tcap):.4f} (parent 0.1471)")
    print(f"M_th(300K)     = {M_threshold(T300):.4f} (parent 0.125)")
    print(f"G_CW(0.7,353K) = {regen_cw_gain_dB(0.7, Tcap):+.2f} dB (parent +9.66)")
    print(f"loss(300K)     = {loss_dB_per_half_lambda(T300):.3f} dB (parent 2.171)")
    print(f"loss(353K)     = {loss_dB_per_half_lambda(Tcap):.3f} dB (parent 2.555)")
