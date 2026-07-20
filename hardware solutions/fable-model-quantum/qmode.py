# -*- coding: utf-8 -*-
"""Quantization of the DS cavity mode: how many quanta is a logic pulse?

The hydrodynamic energy of the acoustic standing mode with fractional density
amplitude eps = (delta n)/n over cell area A is

    E(eps) = (1/4) n m* s^2 eps^2 A                                   (Q1)

(kinetic + potential of the shallow-water mode, spatially averaged standing
wave; energy peaks all-potential at the turning instant). Setting E = hbar*w0
defines the single-plasmon amplitude eps_1, and the quantum content of any
classical swing follows:

    N(eps) = (eps/eps_1)^2,   eps_1 = sqrt(4 hbar w0 / (n m* s^2 A))  (Q2)

Anchored to the parent chain's nonlinear solve: the 1-dB gain-compression knee
(eps ~ 1%) sits at ~38 quanta, the hard rail (eps ~ 10%) at ~3.8e3. Logic
pulses on this fabric are MESOSCOPIC — tens to thousands of plasmons — which
is what makes the quantum noise treatment of qnoise/qerrors quantitative
rather than academic.

This module also owns the honest no-go: the single-plasmon Kerr shift

    chi ~ c_K * w0 * eps_1^2                                          (Q3)

(c_K an O(1) hydrodynamic nonlinearity coefficient, carried as the band
0.05-1 with 0.25 representative — certification belongs to the
Boltzmann-Maxwell tier). chi/kappa ~ 1e-5 in the 576-nm cell: photon-blockade
(gate-model) quantum logic misses by five orders of magnitude. The blockade
area solves chi = kappa AT FIXED omega_0 — a bookkeeping scale, not a
device: a ~3-nm structure cannot host a 1-THz quarter-wave mode at all (its
resonance would sit near 200 THz), so blockade is excluded on this fabric in
two independent ways (magnitude and geometry). This is consistent with
Gullans et al. [PRL 111, 247401 (2013)], whose blockade proposals live at
few-nm scale and correspondingly higher (mid-infrared) frequencies.
"""
import math
import qconstants as C

A_CELL = C.cell_area          # m^2
C_KERR = 0.25                 # representative hydrodynamic Kerr coefficient
C_KERR_BAND = (0.05, 1.0)

EPS_KNEE = 0.01               # 1-dB compression swing (parent solver)
EPS_RAIL = 0.10               # ~14-dB compression, the restoring rail


def mode_energy_scale(n=C.n_op, A=A_CELL):
    """n m* s^2 A — the classical energy scale of the cell mode (J)."""
    return n * C.m_star(n) * C.plasmon_speed(n) ** 2 * A


def eps_one(n=C.n_op, A=A_CELL):
    """Single-plasmon fractional density amplitude, Eq. (Q2)."""
    return math.sqrt(4 * C.E_quantum / mode_energy_scale(n, A))


def N_of_eps(eps, n=C.n_op, A=A_CELL):
    """Quanta in a classical swing eps."""
    return (eps / eps_one(n, A)) ** 2


def kerr_chi(c_K=C_KERR, n=C.n_op, A=A_CELL):
    """Single-plasmon Kerr (self-phase) rate chi, Eq. (Q3), rad/s."""
    return c_K * C.w0 * eps_one(n, A) ** 2


def chi_over_kappa(T=C.T300, c_K=C_KERR):
    """Blockade figure of merit; kappa = 1/tau_q is the energy decay rate."""
    return kerr_chi(c_K) * C.tau_q(T)


def blockade_area(T=C.T300, c_K=C_KERR):
    """Area at which chi = kappa at FIXED omega_0 (chi ~ 1/A): the size scale
    gate-model quantum plasmonics would demand. A bookkeeping scale only —
    no quarter-wave cell of this size supports a 1-THz mode (see module
    docstring)."""
    return A_CELL * chi_over_kappa(T, c_K)


def cross_kerr_phase(N_ctrl, t_int, c_K=C_KERR):
    """Cross-phase (rad) written by N_ctrl plasmons over interaction time t_int
    (cross-Kerr rate = 2 chi)."""
    return 2 * kerr_chi(c_K) * N_ctrl * t_int


def homodyne_phase_resolution(N_probe, T=C.T300):
    """Single-shot homodyne phase std for a coherent probe of N quanta against
    the thermal+vacuum floor: dphi ~ sqrt(nbar + 1/2) / (sqrt(2) sqrt(N))."""
    return math.sqrt(C.nbar(T) + 0.5) / math.sqrt(2 * N_probe)


if __name__ == "__main__":
    e1 = eps_one()
    print(f"mode energy scale n m* s^2 A = {mode_energy_scale():.3e} J")
    print(f"eps_1 (one plasmon)          = {e1:.3e}  ({e1*100:.3f} % swing)")
    print(f"N at 1-dB knee (1% swing)    = {N_of_eps(EPS_KNEE):.1f} quanta")
    print(f"N at rail (10% swing)        = {N_of_eps(EPS_RAIL):.0f} quanta")
    print(f"pulse energy at knee         = {N_of_eps(EPS_KNEE)*C.E_quantum*1e18:.3f} aJ")
    print(f"pulse energy at rail         = {N_of_eps(EPS_RAIL)*C.E_quantum*1e18:.2f} aJ")
    chi = kerr_chi()
    print(f"chi/2pi (c_K=0.25)           = {chi/(2*math.pi)/1e6:.2f} MHz "
          f"(band {kerr_chi(C_KERR_BAND[0])/(2*math.pi)/1e6:.2f}-"
          f"{kerr_chi(C_KERR_BAND[1])/(2*math.pi)/1e6:.1f} MHz)")
    print(f"chi/kappa at 300 K           = {chi_over_kappa(300):.2e}")
    print(f"chi/kappa at 4 K (tau sat.)  = {chi_over_kappa(4):.2e}")
    ab = blockade_area(4)
    print(f"blockade cell area (4 K)     = {ab:.2e} m^2  (~{math.sqrt(ab)*1e9:.1f} nm square)")
    # mesoscopic cross-Kerr: is the interaction phase visible in one shot?
    N = 400
    tint = 2e-12
    phi = cross_kerr_phase(N, tint)
    dphi = homodyne_phase_resolution(N, 4)
    print(f"cross-Kerr phase (N=400, 2 ps) = {phi*1e3:.2f} mrad")
    print(f"homodyne 1-shot resolution 4K  = {dphi*1e3:.1f} mrad "
          f"-> {(dphi/phi)**2:.0f} shots to resolve")
