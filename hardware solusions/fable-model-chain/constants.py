# -*- coding: utf-8 -*-
"""Physical constants and the Fable-Computer operating point.

All SI unless noted. Carrier densities are stored in m^-2 (multiply cm^-2 by 1e4).
This module is the single source of truth for every parameter the chain consumes;
Sections 4 and 7-8 of the manuscript quote values that originate here.
"""
import math

# --- fundamental constants (CODATA) ---
e = 1.602176634e-19        # C
hbar = 1.054571817e-34     # J s
kB = 1.380649e-23          # J/K
eps0 = 8.8541878128e-12    # F/m

# --- graphene / stack material parameters ---
vF = 1.0e6                 # m/s, Fermi velocity
# Out-of-plane hBN permittivity. Literature spans ~3.0-3.8; we report the band
# eps_z in [3.3, 3.8] and adopt eps_z = 3.32 as the design point, which places the
# plasmon speed at the top of the band (s_design = 2.33e6 m/s, Eq. 1).
eps_z = 3.32               # design value (top of the reported band)
eps_z_band = (3.3, 3.8)
d = 10e-9                  # m, gate-graphene hBN spacer thickness
f_OP = 47e12              # Hz, graphene optical-phonon frequency (~0.2 eV / h)
w_OP = 2 * math.pi * f_OP  # rad/s

# --- operating point ---
n_op_cm2 = 1.0e12          # cm^-2, working carrier density
n_op = n_op_cm2 * 1e4      # m^-2
f0 = 1.0e12               # Hz, carrier
w0 = 2 * math.pi * f0      # rad/s
T300 = 300.0               # K
Tcap = 353.0               # K (80 C)
tau_300 = 1.0e-12          # s, momentum-relaxation lifetime at 300 K (best-case)

# clock / pipeline
f_sym = 0.25e12            # Hz, baseline symbol (clock) rate
slot = 1.0 / f_sym         # s (4 ps)

# thermal envelope
T_ambient = 25.0           # C
T_max = 80.0               # C
dT_budget = T_max - T_ambient   # 55 K
R_th = 0.10                # K cm^2 / W, effective stack thermal resistance
fill_worst = 1.0 / 3.0     # worst-case active-area fill

# geometry / yield
cell_area_cm2 = 1e-8       # cm^2 per biased cell (~576 nm x ~1.7 um effective)


def tau(T):
    """Momentum-relaxation time, phonon-limited 1/T scaling above ~150 K."""
    return tau_300 * (T300 / T)


def E_F(n=n_op):
    """Fermi energy of graphene at sheet density n (m^-2)."""
    kF = math.sqrt(math.pi * n)
    return hbar * vF * kF


def k_F(n=n_op):
    return math.sqrt(math.pi * n)


def m_star(n=n_op):
    """Cyclotron/transport effective mass E_F / vF^2 for massless carriers."""
    return E_F(n) / vF**2


def v_sat(n=n_op):
    """Density-dependent saturation velocity, optical-phonon limited: v_sat = w_OP/k_F."""
    return w_OP / k_F(n)


if __name__ == "__main__":
    print(f"E_F(n_op)      = {E_F()/e*1e3:.1f} meV   (manuscript 117 meV)")
    print(f"k_F(n_op)      = {k_F():.3e} m^-1")
    print(f"m*(n_op)       = {m_star():.3e} kg")
    print(f"tau(300K)      = {tau(T300)*1e12:.2f} ps")
    print(f"tau(353K)      = {tau(Tcap)*1e12:.3f} ps")
    print(f"Q(300K)        = {w0*tau(T300):.2f}  (manuscript 6.3)")
    print(f"Q(353K)        = {w0*tau(Tcap):.2f}  (manuscript 5.3)")
