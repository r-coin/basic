# -*- coding: utf-8 -*-
"""Driver: recompute every quantitative claim of the Part-II manuscript and
write results.json. Mirrors fable-model-chain/run_all.py.

    python run_all.py --json
"""
import json
import math
import sys
import qconstants as C
import qmode
import qnoise as Q
import qdecode
import qmac
import qerrors
import qlindblad


def main():
    R = {}

    # --- quantum scales ---
    R["hbar_w0_meV"] = C.E_quantum / C.e * 1e3
    R["T_Q_K"] = C.T_Q
    for T in (4, 20, 48, 77, 150, 300, 353):
        R[f"nbar_{T}K"] = C.nbar(T)
    R["Q_300K"] = C.Q(300)
    R["Q_saturated"] = C.Q(4)

    # --- mode quantization ---
    R["eps_one"] = qmode.eps_one()
    R["N_knee"] = qmode.N_of_eps(qmode.EPS_KNEE)
    R["N_rail"] = qmode.N_of_eps(qmode.EPS_RAIL)
    R["pulse_energy_knee_aJ"] = qmode.N_of_eps(qmode.EPS_KNEE) * C.E_quantum * 1e18
    R["pulse_energy_rail_aJ"] = qmode.N_of_eps(qmode.EPS_RAIL) * C.E_quantum * 1e18

    # --- Kerr no-go ---
    R["chi_over_2pi_MHz"] = qmode.kerr_chi() / (2 * math.pi) / 1e6
    R["chi_over_kappa_300K"] = qmode.chi_over_kappa(300)
    R["chi_over_kappa_4K"] = qmode.chi_over_kappa(4)
    R["blockade_cell_nm"] = math.sqrt(qmode.blockade_area(4)) * 1e9
    R["cross_kerr_phase_mrad_N400_2ps"] = qmode.cross_kerr_phase(400, 2e-12) * 1e3
    R["cross_kerr_shots_to_resolve_4K"] = (
        qmode.homodyne_phase_resolution(400, 4)
        / qmode.cross_kerr_phase(400, 2e-12)) ** 2

    # --- decoder ---
    R["decoder_cells"] = qdecode.CELL_COUNT
    R["F_dec_300K_dB"] = 10 * math.log10(qdecode.F_dec(300))
    R["decoder_truth_table_ok"] = all(
        2 * b1 + b0 == A + B for A, B, lv, b1, b0 in qdecode.truth_table())

    # --- QMAC-1 at the default operating point ---
    N_op = 400
    R["N_op_default"] = N_op
    R["error_table"] = qerrors.table()
    R["preamp_variant_2bit_300K"] = qmac.error_2bit(300, N_op=19, preamp=True)
    R["preamp_variant_2bit_4K"] = qmac.error_2bit(4, N_op=19, preamp=True)
    for tgt, name in ((1e-3, "1e-3"), (1e-6, "1e-6"), (1e-9, "1e-9")):
        R[f"T_2bit_below_{name}"] = qerrors.threshold_temperature(tgt, mode="2bit")
        R[f"T_1bit_below_{name}"] = qerrors.threshold_temperature(tgt, mode="1bit")
    R["quantum_floor"] = qerrors.quantum_floor()
    R["quantum_floor_N800_2bit"] = qmac.error_2bit(4, N_op=800)
    R["launch20dB_wall_77K"] = qmac.error_2bit(77, launch_snr_dB=20)
    R["launch20dB_wall_4K"] = qmac.error_2bit(4, launch_snr_dB=20)
    R["avg16_2bit_300K"] = qmac.error_2bit(300, n_avg=16)
    R["avg16_2bit_353K"] = qmac.error_2bit(353, n_avg=16)
    R["junction_sensitivity_3dB"] = qerrors.sensitivity_junction()
    R["weighted_mac_rms_LSB_77K"] = qmac.weighted_mac_demo()
    R["classical_BER_300K"] = qerrors.classical_ber(300)
    R["classical_BER_353K"] = qerrors.classical_ber(353)

    # --- verification numerics ---
    xn, xt, Vn, Vt = qlindblad.check_loss_thermal()
    R["verify_loss_V_rel_err"] = abs(Vn - Vt) / Vt
    xn, xt, Vn, Vt = qlindblad.check_amplifier()
    R["verify_amp_vac_V_rel_err"] = abs(Vn - Vt) / Vt
    xn, xt, Vn, Vt = qlindblad.check_amplifier(nbar_i=1.0)
    R["verify_amp_thermal_V_rel_err"] = abs(Vn - Vt) / Vt
    lv = [0.0, 10.0, 20.0]
    R["verify_mc_symbol_err"] = {
        "analytic": Q.symbol_error(lv, [0.25, 0.5, 0.25], 4.0),
        "mc": Q.verify_mc(lv, [0.25, 0.5, 0.25], 4.0)}

    print("=" * 64)
    print("FABLE COMPUTER Part II - quantum extension, recomputed results")
    print("=" * 64)
    for k, v in R.items():
        if isinstance(v, float):
            print(f"{k:35s} = {v:.4g}")
        elif isinstance(v, (int, bool)):
            print(f"{k:35s} = {v}")
    print("\nerror table (T, nbar, classical, q2bit, q1bit, q2bit_avg16, ENOB):")
    for r in R["error_table"]:
        print(f"  {r['T_K']:>4.0f} K  {r['nbar']:8.3f}  {r['classical_BER']:9.3e}"
              f"  {r['q2bit']:9.3e}  {r['q1bit']:9.3e}"
              f"  {r['q2bit_avg16']:10.3e}  {r['enob']:5.2f}")

    if "--json" in sys.argv:
        with open("results.json", "w") as f:
            json.dump(R, f, indent=2, default=float)
        print("\nwrote results.json")


if __name__ == "__main__":
    main()
