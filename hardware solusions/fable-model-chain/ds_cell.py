# -*- coding: utf-8 -*-
"""Dyakonov-Shur cell analytics: Eqs (1)-(4) and the Pass-2 single-transit gain.

Everything here is closed-form. The nonlinear regenerative behaviour (Pass 3-4)
lives in solver.py; this module supplies the speeds, thresholds and per-cell
loss budget those solvers are referenced against.
"""
import math
import constants as C


def plasmon_speed(n=C.n_op, d=C.d, eps_z=C.eps_z):
    """Eq (1): gated acoustic-plasmon speed (local-RPA, Dirac)."""
    kF = math.sqrt(math.pi * n)
    g = 2 * C.e**2 * kF * d / (math.pi * C.eps0 * eps_z * C.hbar * C.vF)
    return C.vF * (1 + g) / math.sqrt(1 + 2 * g)


def cell_length(s, f0=C.f0, M=0.0):
    """Eq (4) solved for L: drift-corrected quarter-wave cavity."""
    return (s / (4 * f0)) * (1 - M**2)


def M_threshold(L, s, tau):
    """Eq (3) exact form from the small-M expansion: M_th = L/(2 s tau).
    The compact identity M_th = 1/(8 f0 tau) follows when L = s/(4 f0)."""
    return L / (2 * s * tau)


def ds_increment(M, L, s, tau):
    """Eq (2): small-signal cavity increment omega'' (1/s). >0 = self-oscillation."""
    v0 = M * s
    drive = ((s**2 - v0**2) / (2 * s * L)) * math.log((s + v0) / (s - v0))
    return drive - 1.0 / (2 * tau)


def passive_loss_dB_per_half_lambda(tau, f0=C.f0):
    """Per-gate loss convention: one lambda_p/2 of propagation.
    Power decays as exp(-t/tau); a half-wavelength takes t = 1/(2 f0)."""
    t = 1.0 / (2 * f0)
    return 10 * math.log10(math.exp(t / tau))


def single_transit_net_gain_dB(n, T, M_over_Mth=0.7):
    """Pass 2: analytic single-transit (quarter-wave) net gain at density n, temp T.
    Net = (DS energy increment over one transit) - (passive transit loss).
    Returned in dB; transit time = L/s = 1/(4 f0)."""
    s = plasmon_speed(n)
    tau = C.tau(T)
    L = cell_length(s, M=0.0)
    Mth = M_threshold(L, s, tau)
    M = M_over_Mth * Mth
    omega_pp = ds_increment(M, L, s, tau)          # net amplitude rate (gain - loss)
    t_transit = L / s                               # quarter-wave single transit
    # amplitude grows as exp(omega_pp * t); power gain = exp(2 omega_pp t)
    return 10 * math.log10(math.exp(2 * omega_pp * t_transit))


if __name__ == "__main__":
    s = plasmon_speed()
    print(f"s(n_op)        = {s:.3e} m/s   (manuscript 2.2-2.3e6)")
    print(f"s/vF           = {s/C.vF:.2f}        (Landau-damping margin)")
    L0 = cell_length(s, M=0.0)
    print(f"L (M=0)        = {L0*1e9:.1f} nm     (zero-drift quarter wave)")
    for T, lbl in [(C.T300, "300K"), (C.Tcap, "353K")]:
        tau = C.tau(T)
        Mth = M_threshold(L0, s, tau)
        Mth_id = 1 / (8 * C.f0 * tau)
        M = 0.7 * Mth
        L_op = cell_length(s, M=M)
        print(f"  {lbl}: M_th={Mth:.3f} (identity {Mth_id:.3f}); "
              f"L(0.7 M_th)={L_op*1e9:.1f} nm; v0={M*s:.3e} m/s; M={M:.3f}")
    print(f"per-gate loss  300K = {passive_loss_dB_per_half_lambda(C.tau(C.T300)):.2f} dB "
          f"(manuscript 4.3/2 .. printed as 4.3 per lambda_p)")
    print(f"per-gate loss  353K = {passive_loss_dB_per_half_lambda(C.tau(C.Tcap)):.2f} dB")
    print("Pass-2 single-transit net gain (353K, 0.7 M_th):")
    for n_cm2 in [1e11, 3e11, 1e12, 1e13]:
        g = single_transit_net_gain_dB(n_cm2 * 1e4, C.Tcap)
        print(f"   n={n_cm2:.0e} cm^-2 : {g:+.2f} dB")
