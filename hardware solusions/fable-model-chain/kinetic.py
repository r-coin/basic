# -*- coding: utf-8 -*-
"""Pass 5 (kinetic validity) — the hydrodynamic-tier self-check.

The hydrodynamic equations behind Eq. (2) and the Pass-3 solver assume fast
electron-electron collisions, omega*tau_ee << 1. The standard graphene estimate
1/tau_ee = (kB T)^2/(hbar E_F) gives tau_ee, hence omega*tau_ee, the viscous
correction, and the temperature at which the description expires.
"""
import math
import constants as C


def tau_ee(T=C.T300, EF=None):
    if EF is None:
        EF = C.E_F()
    return C.hbar * EF / (C.kB * T) ** 2


def omega_tau_ee(T=C.T300, f0=C.f0):
    return 2 * math.pi * f0 * tau_ee(T)


def viscous_fraction(T=C.T300):
    """nu q^2 / (1/tau): kinematic viscosity nu = vF^2 tau_ee / 4, q = 2 pi / lambda_p."""
    nu = C.vF ** 2 * tau_ee(T) / 4.0
    from ds_cell import plasmon_speed
    s = plasmon_speed()
    lam = s / C.f0
    q = 2 * math.pi / lam
    return nu * q ** 2 * C.tau(T)


def mfp_ratio(T=C.Tcap):
    """L / l_mfp with l_mfp = vF * tau (momentum-relaxation mean free path)."""
    from ds_cell import plasmon_speed, cell_length
    s = plasmon_speed()
    L = cell_length(s)
    return L / (C.vF * C.tau(T))


def validity_temperature(target=1.0):
    """Temperature at which omega*tau_ee reaches `target` (hydro expires ~1).
    tau_ee ~ 1/T^2 so omega*tau_ee grows as cooling; solve for T below 300 K."""
    base = omega_tau_ee(C.T300)
    # omega*tau_ee(T) = base * (300/T)^2 ; set = target
    return C.T300 * math.sqrt(base / target)


if __name__ == "__main__":
    print(f"tau_ee(300K)     = {tau_ee()*1e15:.0f} fs   (manuscript 115 fs)")
    print(f"omega*tau_ee     = {omega_tau_ee():.2f}     (manuscript 0.7)")
    print(f"viscous fraction = {viscous_fraction()*100:.0f}% of 1/tau   (manuscript ~20%)")
    from ds_cell import plasmon_speed, cell_length
    L = cell_length(plasmon_speed())
    print(f"l_ee/L           = {C.vF*tau_ee()/L:.2f}   (manuscript ~0.2)")
    print(f"L/l_mfp (353K)   = {mfp_ratio():.2f}   (manuscript 0.6-0.7)")
    print(f"hydro expires at = {validity_temperature():.0f} K   (manuscript ~250 K)")
