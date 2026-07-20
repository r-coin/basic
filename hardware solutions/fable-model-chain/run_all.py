# -*- coding: utf-8 -*-
"""Run the full five-pass chain and print/emit every manuscript number.

    python3 run_all.py            # prints the reconciliation table
    python3 run_all.py --json     # also writes results.json
"""
import json
import math
import sys
import numpy as np

import constants as C
import ds_cell as DS
import regen
import cell as CELL
import thermal as TH
import noise as NO
import disorder as DIS
import kinetic as KIN
import coupled as CP
import solver as SOL

s = DS.plasmon_speed(); tau353 = C.tau(C.Tcap); tau300 = C.tau(C.T300)
L = DS.cell_length(s); Mth = DS.M_threshold(L, s, tau353)
v0 = 0.7 * Mth * s


def measure_Mth_num():
    xs = np.linspace(1.05, 1.30, 8)
    g = [SOL.growth_rate(x * Mth, n_roundtrips=90) for x in xs]
    for i in range(len(xs) - 1):
        if g[i] < 0 <= g[i + 1]:
            xz = xs[i] + (xs[i + 1] - xs[i]) * (-g[i]) / (g[i + 1] - g[i])
            return xz * Mth
    return float("nan")


def measure_pulse_gain(Mthn):
    def peak(M):
        r = SOL.run(M, drive_kind="pulse", drive_amp=3e-3, n_roundtrips=240)
        c, t, f0n = r["cav"], r["t"], r["f0_n"]; repT = 1 / (0.25 * f0n)
        pk = [np.max(c[(t >= k * repT) & (t < (k + 1) * repT)])
              for k in range(1, int(t[-1] / repT)) if ((t >= k * repT) & (t < (k + 1) * repT)).sum() > 5]
        pk = np.array(pk); return pk[len(pk) // 2:].mean()
    return 20 * math.log10(peak(0.7 * Mthn) / peak(1e-9))


def main():
    G_cw = regen.cw_net_gain_dB(0.7)
    G_pow = 10 ** (G_cw / 10)
    Mthn = measure_Mth_num()
    g_pulse = measure_pulse_gain(Mthn)
    P_op = TH.joule_kW_cm2(C.n_op, v0, tau353)
    P_hi = TH.joule_kW_cm2(C.n_op, 0.118 * s, tau353)
    P_block = (P_op * 1e3) * (7 * C.cell_area_cm2) * 1e3
    P_block_hi = (P_hi * 1e3) * (7 * C.cell_area_cm2) * 1e3
    casc = CELL.cascade_per_cell_gain_dB()
    R = {
        "plasmon_speed_design_m_s": s,
        "speed_band_m_s": [DS.plasmon_speed(eps_z=C.eps_z_band[1]), DS.plasmon_speed(eps_z=C.eps_z_band[0])],
        "cell_length_zero_drift_nm": DS.cell_length(s, M=0) * 1e9,
        "cell_length_operating_nm": DS.cell_length(s, M=0.7 * Mth) * 1e9,
        "E_F_meV": C.E_F() / C.e * 1e3,
        "Q_300K": C.w0 * tau300, "Q_353K": C.w0 * tau353,
        "M_th_300K": DS.M_threshold(DS.cell_length(s), s, tau300),
        "M_th_353K": Mth,
        "M_th_num": Mthn,
        "v0_operating_m_s": v0,
        "J_mA_per_um": C.n_op * C.e * v0 * 1e-3 / 1e6 * 1e6 * 1e-3 * 1e3,  # see note
        "per_gate_loss_300K_dB": DS.passive_loss_dB_per_half_lambda(tau300),
        "per_gate_loss_353K_dB": DS.passive_loss_dB_per_half_lambda(tau353),
        "cw_regen_gain_dB_at_0p7": G_cw,
        "pulse_gain_dB_at_0p7_streaming": g_pulse,
        "cascade_per_cell_dB": casc,
        "compression_1pct_dB": CELL.compression_dB(0.01),
        "compression_10pct_dB": CELL.compression_dB(0.10),
        "noise_figure_floor_dB": NO.noise_figure(G_pow),
        "noise_margin_frac": CELL.static_noise_margin(),
        "yield_1e12_literature_pct": 100 * DIS.yield_fraction(1e12, 2.5e9, 4e10),
        "yield_1e12_broad_pct": 100 * DIS.yield_fraction(1e12, 2.5e9, 3e11),
        "yield_3e11_literature_pct": 100 * DIS.yield_fraction(3e11, 2.5e9, 4e10),
        "yield_1e11_literature_pct": 100 * DIS.yield_fraction(1e11, 2.5e9, 4e10),
        "dissipation_operating_kW_cm2": P_op,
        "dissipation_upper_kW_cm2": P_hi,
        "block_power_mW": [P_block, P_block_hi],
        "energy_per_add_fJ": [TH.energy_per_add(P_block), TH.energy_per_add(P_block_hi)],
        "dT_fill_third_operating_K": TH.dT(P_op, C.fill_worst),
        "dT_fill_third_upper_K": TH.dT(P_hi, C.fill_worst),
        "dT_block_K": TH.dT(P_op, 0.04),
        "pass1_corner_kW_cm2": TH.joule_kW_cm2(1e13 * 1e4, C.v_sat(1e13 * 1e4), tau353),
        "tau_ee_fs_300K": KIN.tau_ee() * 1e15,
        "omega_tau_ee": KIN.omega_tau_ee(),
        "viscous_fraction": KIN.viscous_fraction(),
        "L_over_mfp_353K": KIN.mfp_ratio(),
        "hydro_expiry_K": KIN.validity_temperature(),
        "coupled_max_growth_0p7": float(CP.growth_vs_clamp(0.7)[1].max()),
        "coupled_max_growth_0p9": float(CP.growth_vs_clamp(0.9)[1].max()),
        "vsat_over_s_1e13": C.v_sat(1e13 * 1e4) / DS.plasmon_speed(n=1e13 * 1e4),
    }
    # correct J: n[m^-2]*e*v0 [A/m]; /1e3 -> mA; *1e-6 m/um -> per um
    R["J_mA_per_um"] = C.n_op * C.e * v0 / 1e3 * 1e-6 * 1e3  # A/m ->mA/um
    R["J_mA_per_um"] = (C.n_op * C.e * v0) * 1e-6 / 1e-3      # (A/m)*(1e-6 m/um)/(1e-3 A/mA)

    print("=" * 64)
    print("FABLE COMPUTER — five-pass model chain, recomputed results")
    print("=" * 64)
    for k, val in R.items():
        if isinstance(val, dict):
            print(f"{k}:")
            for kk, vv in val.items():
                print(f"    junction {kk:+d} dB : {vv:+.2f} dB")
        elif isinstance(val, list):
            print(f"{k:34s} = " + ", ".join(f"{x:.3g}" for x in val))
        else:
            print(f"{k:34s} = {val:.4g}")
    if "--json" in sys.argv:
        with open("results.json", "w") as f:
            json.dump(R, f, indent=2)
        print("\nwrote results.json")


if __name__ == "__main__":
    main()
