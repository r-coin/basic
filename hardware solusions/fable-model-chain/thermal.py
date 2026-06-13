# -*- coding: utf-8 -*-
"""Pass 1 (thermal no-go) and Section 8 (energy + thermal envelope).

Joule dissipation density P = n m* v0^2 / tau (W/m^2 sheet), spread through an
effective stack resistance R_th to a temperature rise dT = q_areal * R_th, with
q_areal = P * fill. The high-density corner (n=1e13, drift at v_sat) is the
brute-force no-go; the operating point (n=1e12, v0=2.4e5) closes the 55 K budget
with margin at fill 1/3.
"""
import math
import constants as C


def joule_density(n, v0, tau):
    """Sheet dissipation density (W/m^2)."""
    return n * C.m_star(n) * v0 ** 2 / tau


def joule_kW_cm2(n, v0, tau):
    return joule_density(n, v0, tau) * 1e-3 / 1e4    # W/m^2 -> kW/cm^2


def dT(P_kW_cm2, fill):
    """Steady temperature rise (K): q_areal[W/cm^2] * R_th[K cm^2 / W]."""
    return P_kW_cm2 * 1e3 * fill * C.R_th


def duty_ceiling(P_kW_cm2, fill):
    """Max duty cycle that keeps dT within the budget."""
    dT_full = dT(P_kW_cm2, fill)
    return min(1.0, C.dT_budget / dT_full)


def energy_per_add(P_mW, slot=C.slot):
    """fJ per addition: power * slot time."""
    return P_mW * 1e-3 * slot * 1e15


if __name__ == "__main__":
    tau = C.tau(C.Tcap)
    # operating point
    s = 2.33e6
    v0 = 0.103 * s
    P_op = joule_kW_cm2(C.n_op, v0, tau)
    print(f"operating-point dissipation = {P_op:.2f} kW/cm^2  (v0={v0:.2e} m/s)")
    # upper (solver-class) bias bound: M/M_th,num=0.7 -> M=0.118
    v0b = 0.118 * s
    P_hi = joule_kW_cm2(C.n_op, v0b, tau)
    print(f"upper-bias dissipation      = {P_hi:.2f} kW/cm^2  (v0={v0b:.2e} m/s)")
    for tag, P in [("operating", P_op), ("upper bias", P_hi)]:
        print(f"  {tag}: dT(fill 1/3)={dT(P, C.fill_worst):.1f} K  "
              f"(budget {C.dT_budget:.0f} K, margin {100*(1-dT(P,C.fill_worst)/C.dT_budget):+.0f}%)")
    # half-adder block: 7 biased cells, ~4% fill
    # power = P[kW/cm2]*1e3 [W/cm2] * area[cm2] -> W ; *1e3 -> mW
    P_block_mW = (P_op * 1e3) * (7 * C.cell_area_cm2) * 1e3
    P_block_hi = (P_hi * 1e3) * (7 * C.cell_area_cm2) * 1e3
    print(f"half-adder block power       = {P_block_mW:.3f}-{P_block_hi:.3f} mW")
    print(f"energy per addition          = {energy_per_add(P_block_mW):.2f}-{energy_per_add(P_block_hi):.2f} fJ")
    print(f"block dT (~4% fill)          = {dT(P_op, 0.04):.1f} K")
    # Pass-1 high-density corner: n=1e13, drift at v_sat
    n13 = 1e13 * 1e4
    vsat13 = C.v_sat(n13)
    P13 = joule_kW_cm2(n13, vsat13, tau)
    print(f"\nPass-1 corner n=1e13 at v_sat: {P13:.0f} kW/cm^2; "
          f"duty ceiling (fill 1/3) = {100*duty_ceiling(P13, C.fill_worst):.2f}%, "
          f"(full fill) = {100*duty_ceiling(P13, 1.0):.2f}%")
    print(f"headroom op->corner = {P13/P_op:.0f}x")
