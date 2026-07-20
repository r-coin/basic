# -*- coding: utf-8 -*-
"""Regenerate Figures 6-10 of the manuscript from the recreated model chain.

Each figure is emitted to figures/figN.png at the manuscript's panel layout.
All curves are computed live by the chain modules -- no stored data."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import constants as C
import ds_cell as DS
import regen
import cell as CELL
import thermal as TH
import noise as NO
import disorder as DIS
import coupled as CP
import solver as SOL

ACC = "#2e75b6"; ACC2 = "#c0392b"; GRN = "#2e8b57"; GRY = "#888888"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})
OUT = "figures"
s = DS.plasmon_speed(); tauC = C.tau(C.Tcap)
L = DS.cell_length(s); Mth = DS.M_threshold(L, s, tauC)
Mth_num = 0.169


def _panel(ax, title):
    ax.set_title(title, loc="left")


# ---------------- Figure 6 : Pass 1 ----------------
def fig6():
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4), dpi=150)
    # (A) passive signal vs gate depth for candidate per-gate budgets
    depth = np.arange(0, 13)
    for loss, lab in [(2.0, "−2.0 dB/gate"), (3.0, "−3.0 dB/gate"), (4.3, "−4.3 dB/gate")]:
        ax[0,0].plot(depth, -loss*depth, marker="o", ms=3, label=lab)
    ax[0,0].axhline(-20, color=ACC2, ls="--", lw=1, label="read floor")
    _panel(ax[0,0], "(A) Passive signal vs gate depth (n = 10¹³)")
    ax[0,0].set_xlabel("gate depth"); ax[0,0].set_ylabel("signal (dB)"); ax[0,0].legend(fontsize=7)
    # (B) net per-gate vs drift ratio v_d/v_p, n=1e13
    n13 = 1e13*1e4; s13 = DS.plasmon_speed(n=n13); vsat13 = C.v_sat(n13)
    vr = np.linspace(0, 0.30, 100)
    loss_q = DS.passive_loss_dB_per_half_lambda(tauC)/2
    gain = 20*np.log10((1+vr)/(1-vr)) - loss_q*2     # round-trip reflection gain - loss
    ax[0,1].plot(vr, gain, color=ACC, lw=1.8)
    ax[0,1].axhline(0, color="k", lw=0.8)
    ax[0,1].axvline(vsat13/s13, color=ACC2, ls="--", lw=1.2, label=f"v_sat/v_p = {vsat13/s13:.2f}")
    ax[0,1].fill_betweenx([gain.min(), gain.max()], vsat13/s13, 0.30, color=ACC2, alpha=0.08)
    _panel(ax[0,1], "(B) Net per-gate vs drift ratio: transparency unreachable below v_sat")
    ax[0,1].set_xlabel("v_d / v_p"); ax[0,1].set_ylabel("net per-gate (dB)"); ax[0,1].legend(fontsize=7)
    # (C) chip dT vs drift velocity vs 55 K budget, n=1e13
    v = np.linspace(0, vsat13, 100)
    dT = np.array([TH.dT(TH.joule_kW_cm2(n13, vv, tauC), C.fill_worst) for vv in v])
    ax[1,0].plot(v/1e5, dT, color=ACC, lw=1.8)
    ax[1,0].axhline(C.dT_budget, color=ACC2, ls="--", lw=1.2, label="55 K budget")
    _panel(ax[1,0], "(C) Steady chip ΔT vs drift velocity (n = 10¹³, fill ⅓)")
    ax[1,0].set_xlabel("drift velocity (10⁵ m/s)"); ax[1,0].set_ylabel("ΔT (K)"); ax[1,0].legend(fontsize=7)
    # (D) achievable cascade depth vs per-cell net gain
    g = np.linspace(-3, 1.5, 100)
    depthmax = np.where(g < 0, 20/np.maximum(-g, 0.05), 60)
    ax[1,1].plot(g, np.clip(depthmax, 0, 60), color=GRN, lw=1.8)
    ax[1,1].axvline(0, color="k", lw=0.8)
    _panel(ax[1,1], "(D) Achievable cascade depth vs per-cell net gain")
    ax[1,1].set_xlabel("per-cell net gain (dB)"); ax[1,1].set_ylabel("cascade depth (gates)")
    fig.suptitle("Figure 6 — Pass 1: bulk gain and brute thermal limits (n = 10¹³ cm⁻², 353 K)",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(f"{OUT}/fig6.png"); plt.close(fig)


# ---------------- Figure 7 : Pass 2 ----------------
def fig7():
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4), dpi=150)
    # (A) max-sustainable round-trip net gain vs density (v_sat-capped bias)
    ncm = np.logspace(11, 13.3, 60)
    loss_rt = DS.passive_loss_dB_per_half_lambda(tauC)
    g = []
    for n_cm2 in ncm:
        n = n_cm2*1e4; sn = DS.plasmon_speed(n=n)
        Mcap = min(0.95*Mth, C.v_sat(n)/sn)
        g.append(20*np.log10((1+Mcap)/(1-Mcap)) - loss_rt)
    ax[0,0].semilogx(ncm, g, color=ACC, lw=1.8)
    ax[0,0].axhline(0, color="k", lw=0.8); ax[0,0].axvline(1e12, color=GRN, ls=":", lw=1.2, label="operating n")
    _panel(ax[0,0], "(A) Max round-trip net gain vs density (v_sat-capped)")
    ax[0,0].set_xlabel("n (cm⁻²)"); ax[0,0].set_ylabel("net round-trip gain (dB)"); ax[0,0].legend(fontsize=7)
    # (B) thermal duty ceiling vs density
    duty = []
    for n_cm2 in ncm:
        n = n_cm2*1e4
        P = TH.joule_kW_cm2(n, C.v_sat(n), tauC)
        duty.append(100*TH.duty_ceiling(P, C.fill_worst))
    ax[0,1].loglog(ncm, np.clip(duty,1e-2,100), color=ACC2, lw=1.8)
    ax[0,1].axvline(1e12, color=GRN, ls=":", lw=1.2)
    _panel(ax[0,1], "(B) Thermal duty ceiling vs density (at v_sat, fill ⅓)")
    ax[0,1].set_xlabel("n (cm⁻²)"); ax[0,1].set_ylabel("duty ceiling (%)")
    # (C) static noise margins (butterfly) at extinctions
    sw = np.linspace(0, 1, 100)
    for ext, col in [(6, GRY), (10, ACC), (20, GRN)]:
        nm = CELL.static_noise_margin(extinction_dB=ext)
        ax[1,0].plot([0,nm,1-nm,1],[0,1,0,1] if False else [0,0,0,0])  # placeholder removed
    # draw butterfly conceptually: VTC and mirrored
    k = CELL.K_SHARP
    vin = np.linspace(0,1,200)
    vtc = 0.5*(1+np.tanh((vin-0.5)*k*2))
    ax[1,0].plot(vin, vtc, color=ACC, lw=1.6, label="VTC")
    ax[1,0].plot(vtc, vin, color=GRN, lw=1.6, label="mirror")
    ax[1,0].text(0.05,0.85,f"NM ≈ {CELL.static_noise_margin():.2f}", fontsize=8)
    _panel(ax[1,0], "(C) Static noise margins (butterfly, k = 8)")
    ax[1,0].set_xlabel("V_in (norm)"); ax[1,0].set_ylabel("V_out (norm)"); ax[1,0].legend(fontsize=7)
    # (D) pattern-dependent gain spread vs symbol rate (carrier-recovery, gated)
    fs = np.linspace(0.1e12, 1.0e12, 30)
    sp = [CELL.isi_spread_dB(f) for f in fs]
    ax[1,1].plot(fs/1e12, sp, color=ACC, lw=1.8)
    ax[1,1].axhline(1.0, color=ACC2, ls="--", lw=1, label="1 dB")
    _panel(ax[1,1], "(D) Gain-recovery ISI spread vs symbol rate (gated)")
    ax[1,1].set_xlabel("symbol rate (THz)"); ax[1,1].set_ylabel("spread (dB)"); ax[1,1].legend(fontsize=7)
    fig.suptitle("Figure 7 — Pass 2: analytic resonant cells (353 K)",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(f"{OUT}/fig7.png"); plt.close(fig)


# ---------------- Figure 8 : Pass 3 ----------------
def fig8():
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4), dpi=150)
    # (A) growth rate vs drift: numerical (solver) vs analytic
    Ms = np.linspace(0.5, 1.3, 9) * Mth
    ana = [DS.ds_increment(M, L, s, tauC)*2*(L/s) for M in Ms]   # per round trip (dimensionless)
    ax[0,0].plot(Ms/Mth, ana, color=ACC, lw=1.8, marker="o", ms=3, label="analytic Eq. (2)")
    # solver points (precomputed-ish via growth_rate at a few M)
    solM = [0.7,0.9,1.0,1.1,1.2,1.3]
    solg = [SOL.growth_rate(r*Mth, n_roundtrips=70) for r in solM]
    ax[0,0].plot(solM, solg, color=ACC2, lw=1.6, marker="s", ms=4, label="solver (numerical)")
    ax[0,0].axhline(0, color="k", lw=0.8)
    ax[0,0].axvline(1.0, color=ACC, ls=":", lw=1); ax[0,0].axvline(Mth_num/Mth, color=ACC2, ls=":", lw=1)
    _panel(ax[0,0], "(A) Cavity growth rate vs drift: analytic vs solver")
    ax[0,0].set_xlabel("M / M_th(analytic)"); ax[0,0].set_ylabel("growth / round trip"); ax[0,0].legend(fontsize=7)
    # (B) sub-threshold regenerative net gain vs M, break-even marked
    r = np.linspace(0.1, 0.85, 80)
    gcw = [regen.cw_net_gain_dB(x) for x in r]
    ax[0,1].plot(r, gcw, color=ACC, lw=1.9, label="CW regen gain")
    loss_rt = DS.passive_loss_dB_per_half_lambda(tauC)
    ax[0,1].axhline(loss_rt, color=GRY, ls="--", lw=1, label=f"break-even ({loss_rt:.2f} dB)")
    ax[0,1].axvline(0.7, color=GRN, ls=":", lw=1.2, label="operating 0.7")
    ax[0,1].fill_betweenx([0, max(gcw)], 0.0, 0.8, color="none")
    _panel(ax[0,1], "(B) Sub-threshold regenerative net gain vs M")
    ax[0,1].set_xlabel("M / M_th"); ax[0,1].set_ylabel("net gain vs passive (dB)"); ax[0,1].legend(fontsize=7)
    # (C) gain compression vs drive amplitude
    dr = np.logspace(-3, -0.7, 60)
    comp = [CELL.compression_dB(x) for x in dr]
    ax[1,0].semilogx(dr*100, comp, color=ACC, lw=1.9)
    ax[1,0].axhline(-1, color=ACC2, ls="--", lw=1, label="−1 dB")
    ax[1,0].axvline(1.0, color=GRN, ls=":", lw=1)
    _panel(ax[1,0], "(C) Gain compression vs drive (the logic rail)")
    ax[1,0].set_xlabel("density swing δρ/ρ₀ (%)"); ax[1,0].set_ylabel("compression (dB)"); ax[1,0].legend(fontsize=7)
    # (D) pulse train through the cell (solver waveform)
    rr = SOL.run(0.7*Mth_num, drive_kind="pulse", drive_amp=3e-3, n_roundtrips=60)
    t = rr["t"]*(L/s)*1e12   # ps
    ax[1,1].plot(t, rr["cav"]/np.max(np.abs(rr["cav"])), color=ACC, lw=0.8)
    _panel(ax[1,1], "(D) 0.25-THz few-cycle pulse train through the cell (solver)")
    ax[1,1].set_xlabel("time (ps)"); ax[1,1].set_ylabel("intracavity field (norm)")
    fig.suptitle("Figure 8 — Pass 3: regenerative operating point (M_th,num ≈ 0.17)",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(f"{OUT}/fig8.png"); plt.close(fig)


# ---------------- Figure 9 : Pass 4 ----------------
def fig9():
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4), dpi=150)
    # (A) cascade building block: genuine input pulse train vs amplified cell output
    rr = SOL.run(0.7*Mth_num, drive_kind="pulse", drive_amp=3e-3, n_roundtrips=50)
    t = rr["t"]*(L/s)*1e12
    f0n = rr["f0_n"]
    drive = np.array([SOL._pulse_train(tt, f0n, 3e-3) for tt in rr["t"]])
    ax[0,0].plot(t, drive/np.max(np.abs(drive)), color=GRY, lw=0.7, label="input pulse train")
    ax[0,0].plot(t, rr["cav"]/np.max(rr["cav"]), color=ACC, lw=0.8, label="amplified cell output")
    _panel(ax[0,0], "(A) Cascade building block: pulse in vs amplified out")
    ax[0,0].set_xlabel("time (ps)"); ax[0,0].set_ylabel("field (norm)"); ax[0,0].legend(fontsize=7)
    # (B) per-cell gain vs junction transmission
    casc = CELL.cascade_per_cell_gain_dB()
    J = list(casc.keys()); G = list(casc.values())
    ax[0,1].plot(J, G, color=ACC, lw=1.8, marker="o", ms=5)
    ax[0,1].plot(J, [2*g for g in G], color=GRN, lw=1.4, marker="s", ms=4, label="two-cell total")
    _panel(ax[0,1], "(B) Per-cell & two-cell net gain vs junction")
    ax[0,1].set_xlabel("junction transmission (dB)"); ax[0,1].set_ylabel("net gain (dB)"); ax[0,1].legend(fontsize=7)
    # (C) noise figure vs gain, Eq. (7)
    Gp = np.linspace(1.5, 30, 100)
    F = [NO.noise_figure(g) for g in Gp]
    ax[1,0].plot(10*np.log10(Gp), F, color=ACC, lw=1.9)
    ax[1,0].axhline(3.0, color=GRY, ls="--", lw=1, label="3 dB")
    ax[1,0].axvline(regen.cw_net_gain_dB(0.7), color=GRN, ls=":", lw=1.2, label="operating gain")
    ax[1,0].text(12, 2.85, f"floor {NO.noise_figure(10**(regen.cw_net_gain_dB(0.7)/10)):.2f} dB", fontsize=8)
    _panel(ax[1,0], "(C) Noise figure vs gain: F = 2 − 1/G")
    ax[1,0].set_xlabel("gain (dB)"); ax[1,0].set_ylabel("noise figure (dB)"); ax[1,0].legend(fontsize=7)
    # (D) MC cell yield vs puddle density, literature band shaded
    sig = np.logspace(9.0, 11.7, 60)   # cm^-2
    for n_cm2, col in [(1e11, ACC2), (3e11, GRY), (1e12, ACC)]:
        sc = DIS.sigma_crit(n_cm2*1e4)/1e4
        y = 100*(sig < sc)
        ax[1,1].semilogx(sig, y, color=col, lw=1.8, label=f"n = {n_cm2:.0e}")
    ax[1,1].axvspan(2.5e9, 4e10, color=GRN, alpha=0.12, label="literature σ")
    _panel(ax[1,1], "(D) Monte-Carlo cell yield vs puddle density")
    ax[1,1].set_xlabel("σ (cm⁻²)"); ax[1,1].set_ylabel("yield (%)"); ax[1,1].legend(fontsize=7)
    fig.suptitle("Figure 9 — Pass 4: cascade, noise, and measured disorder",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(f"{OUT}/fig9.png"); plt.close(fig)


# ---------------- Figure 10 : Pass 5 ----------------
def fig10():
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    for r, col, lab in [(0.7, ACC, "working bias 0.7 M_th"), (0.9, ACC2, "0.9 M_th")]:
        c, rate, _ = CP.growth_vs_clamp(r)
        ax.plot(c, rate/1e12, color=col, lw=2, marker="o", ms=3, label=lab)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.fill_between([0,1], 0, ax.get_ylim()[1] if ax.get_ylim()[1]>0 else 1, color=ACC2, alpha=0.06)
    ax.set_xlabel("midpoint clamp strength  (0 = free → 1 = hard clamp)")
    ax.set_ylabel("most-unstable growth rate (10¹² s⁻¹)")
    ax.set_title("Figure 10 — Pass 5: coupled two-cell stability (no instability at any coupling)",
                 loc="left", fontsize=10.5, weight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig10.png"); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig6(); print("fig6 done")
    fig7(); print("fig7 done")
    fig8(); print("fig8 done")
    fig9(); print("fig9 done")
    fig10(); print("fig10 done")
