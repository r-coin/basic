# -*- coding: utf-8 -*-
"""Figures Q1-Q5 for the Part-II manuscript. Writes figures/fig_q*.png.

Q1  quantum content of the fabric: quanta vs swing, thermal occupation vs T,
    the Kerr-blockade gap vs cell size, and the decision-noise budget vs T
Q2  QMAC-1 schematic and the three-level decode at 300 / 77 / 4 K
Q3  error rate vs temperature: quantum-analog 1/2-bit vs classical digital
Q4  per-shot analog precision (ENOB) vs T; error vs N_op at fixed T
Q5  decoder cell map on the gate lattice (parent Figure 5b style)
"""
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

import qconstants as C
import qmode
import qdecode
import qmac
import qerrors

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def fig_q1():
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.6))

    # (A) quanta vs swing
    eps = np.logspace(-3.2, -0.7, 200)
    A = ax[0, 0]
    A.loglog(eps * 100, [qmode.N_of_eps(e) for e in eps], "k-")
    for e, lab in ((qmode.eps_one(), "1 plasmon"),
                   (qmode.EPS_KNEE, f"1-dB knee: {qmode.N_of_eps(0.01):.0f}"),
                   (qmode.EPS_RAIL, f"rail: {qmode.N_of_eps(0.1):.0f}")):
        A.axvline(e * 100, color="C3", ls=":", lw=1)
        A.annotate(lab, (e * 100, 2), rotation=90, fontsize=7.5,
                   ha="right", va="bottom")
    A.set_xlabel("fractional density swing (%)")
    A.set_ylabel("mode quanta N")
    A.set_title("(A) quantum content of a logic swing")

    # (B) thermal occupation
    B = ax[0, 1]
    T = np.linspace(2, 400, 400)
    B.semilogy(T, [max(C.nbar(t), 1e-7) for t in T], "k-")
    B.axvline(C.T_Q, color="C0", ls="--", lw=1)
    B.annotate(f"$T_Q$ = {C.T_Q:.0f} K", (C.T_Q + 8, 2e-4), fontsize=8)
    B.axhline(0.5, color="C3", ls=":", lw=1)
    B.annotate("vacuum 1/2", (250, 0.6), fontsize=7.5, color="C3")
    for t in (77, 300):
        B.plot(t, C.nbar(t), "ko", ms=3)
        B.annotate(f"{C.nbar(t):.2f}", (t * 1.03, C.nbar(t) * 1.2), fontsize=7.5)
    B.set_xlabel("temperature (K)")
    B.set_ylabel(r"thermal occupation $\bar{n}$")
    B.set_title("(B) the 1-THz mode meets its bath")

    # (C) Kerr blockade gap vs cell size
    Cx = ax[1, 0]
    side = np.logspace(math.log10(1e-9), math.log10(2e-6), 200)
    ck = [qmode.kerr_chi(n=C.n_op, A=s * s) * C.tau_q(4) for s in side]
    Cx.loglog(side * 1e9, ck, "k-")
    Cx.axhline(1, color="C3", ls=":", lw=1)
    Cx.annotate("blockade $\\chi=\\kappa$", (2, 2), fontsize=8, color="C3")
    s_cell = math.sqrt(qmode.A_CELL)
    Cx.plot(s_cell * 1e9, qmode.chi_over_kappa(4), "C0o")
    Cx.annotate("this cell\n($10^{-5}$ short)", (s_cell * 1e9 * 0.9,
                qmode.chi_over_kappa(4) * 6), fontsize=8, ha="right")
    sb = math.sqrt(qmode.blockade_area(4))
    Cx.plot(sb * 1e9, 1.0, "C3s")
    Cx.annotate(f"{sb*1e9:.0f} nm", (sb * 1e9 * 1.4, 0.5), fontsize=8)
    Cx.set_xlabel("cell side (nm)   [chi ~ 1/A at fixed n]")
    Cx.set_ylabel(r"$\chi/\kappa$ (4 K)")
    Cx.set_title("(C) gate-model quantum logic: the size gap")

    # (D) decision-noise budget vs T (actual model components at each T)
    D = ax[1, 1]
    T = np.linspace(4, 400, 120)
    vac, thermal, dec_amp, band = [], [], [], []
    for t in T:
        states, xr = qmac.levels_at_decision(float(t))
        V_sig = max(s.V for s in states.values())
        vac.append(0.5)
        thermal.append(max(V_sig - 0.5, 0.0))      # bath mixed in via loss
        dec_amp.append(qdecode.decoder_amp_noise(float(t)))
        band.append(qdecode.threshold_band_variance(xr))
    D.stackplot(T, vac, thermal, dec_amp, band,
                labels=["vacuum", "thermal bath (via loss)",
                        "comparator amp", "threshold band"],
                colors=["#c6dbef", "#fdd0a2", "#c7e9c0", "#dadaeb"])
    D.axvline(C.T_Q, color="C0", ls="--", lw=1)
    D.set_xlabel("temperature (K)")
    D.set_ylabel("decision variance (quanta units)")
    D.legend(loc="upper left", fontsize=7.5)
    D.set_title("(D) what the decision fights, vs T")

    fig.suptitle("Figure Q1. Quantum scales of the plasmon fabric "
                 "(operating point of Part I)", y=1.005)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_q1.png"))
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc="#dce9f5", fontsize=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec="k", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=10, lw=0.9, color="k"))


def fig_q2():
    fig = plt.figure(figsize=(9.5, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.4], hspace=0.32)

    # (A) schematic
    A = fig.add_subplot(gs[0])
    A.set_xlim(0, 14)
    A.set_ylim(0, 5)
    A.axis("off")
    _box(A, 0.2, 3.4, 1.8, 1.0, "launch A\n(comb-locked)")
    _box(A, 0.2, 0.6, 1.8, 1.0, "launch B\n(comb-locked)")
    _box(A, 2.6, 3.4, 1.6, 1.0, "weight $\\sqrt{w}$\n(gate map)", fc="#fff3cd")
    _box(A, 2.6, 0.6, 1.6, 1.0, "weight $\\sqrt{1-w}$\n(gate map)", fc="#fff3cd")
    _box(A, 4.9, 2.0, 1.9, 1.1, "junction\ncombiner\n(interference)", fc="#fde2cf")
    _box(A, 7.4, 2.0, 1.3, 1.1, "fan-out\nsplit")
    _box(A, 9.3, 3.2, 1.6, 0.9, "comparator $t_2$\n(cell, $k{=}16$)", fc="#d5e8d4")
    _box(A, 9.3, 0.9, 1.6, 0.9, "comparator $t_1$\n(cell, $k{=}16$)", fc="#d5e8d4")
    _box(A, 11.5, 3.2, 1.0, 0.9, "buffer", fc="#d5e8d4")
    _box(A, 11.5, 0.9, 1.3, 0.9, "NOT / AND", fc="#d5e8d4")
    A.text(13.3, 3.65, "$b_1$", fontsize=11)
    A.text(13.3, 1.35, "$b_0$", fontsize=11)
    _arrow(A, 2.0, 3.9, 2.6, 3.9); _arrow(A, 2.0, 1.1, 2.6, 1.1)
    _arrow(A, 4.2, 3.9, 4.9, 2.9); _arrow(A, 4.2, 1.1, 4.9, 2.2)
    _arrow(A, 6.8, 2.55, 7.4, 2.55)
    _arrow(A, 8.7, 2.8, 9.3, 3.6); _arrow(A, 8.7, 2.3, 9.3, 1.4)
    _arrow(A, 10.9, 3.65, 11.5, 3.65); _arrow(A, 10.9, 1.35, 11.5, 1.35)
    _arrow(A, 10.9, 3.5, 11.5, 1.6)   # t2 into NOT/AND
    _arrow(A, 12.5, 3.65, 13.2, 3.65); _arrow(A, 12.8, 1.35, 13.2, 1.35)
    A.text(5.85, 1.2, "analog core: passive,\nphase-coherent", fontsize=7.5,
           ha="center", style="italic")
    A.text(10.9, 0.15, "decoder: the Part-I half-adder cells as a flash ADC",
           fontsize=7.5, ha="center", style="italic")
    A.set_title("(A) QMAC-1: two inputs, one interference, one decoded 2-bit word")

    # (B) levels + Gaussians at three temperatures
    temps = [300.0, 77.0, 4.0]
    for i, T in enumerate(temps):
        B = fig.add_subplot(gs[1].subgridspec(1, 3, wspace=0.25)[0, i])
        states, xr = qmac.levels_at_decision(T)
        V = qmac.decision_variance(states, xr, T)
        L0 = states[(0, 0)].x
        L1 = 0.5 * (states[(0, 1)].x + states[(1, 0)].x)
        L2 = states[(1, 1)].x
        xs = np.linspace(-4, L2 + 4 * math.sqrt(V), 500)
        for Lx, pr, lab in ((L0, 0.25, "0"), (L1, 0.5, "1"), (L2, 0.25, "2")):
            B.fill_between(xs, pr * np.exp(-(xs - Lx) ** 2 / (2 * V))
                           / math.sqrt(2 * math.pi * V), alpha=0.55,
                           label=f"level {lab}")
        for th in ((L0 + L1) / 2, (L1 + L2) / 2):
            B.axvline(th, color="k", ls=":", lw=0.9)
        B.set_title(f"{T:.0f} K   (err {qmac.error_2bit(T):.1e})", fontsize=8.5)
        B.set_xlabel("decision quadrature x")
        if i == 0:
            B.set_ylabel("probability density")
        B.set_yticks([])
    fig.suptitle("Figure Q2. The minimal quantum-analog tensor unit and its "
                 "2-bit decode ($N_{op}$ = 400 quanta)", y=0.99)
    fig.savefig(os.path.join(OUT, "fig_q2.png"))
    plt.close(fig)


def fig_q3():
    sw = qerrors.sweep()
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    T = sw["T"]
    FLOOR = 1e-24
    curves = [
        ("classical", "classical digital (rail, Part I)", "k-"),
        ("q1bit", "quantum-analog, 1-bit decode", "C0-"),
        ("q2bit", "quantum-analog, 2-bit decode", "C3-"),
        ("q2bit_avg16", "2-bit, 16-slot averaging", "C3--"),
        ("q2bit_launch20", "2-bit, 20-dB classical launch", "C7-."),
    ]
    for key, lab, sty in curves:
        y = np.clip(sw[key], FLOOR, 1)
        ax.plot(T, y, sty, label=lab, lw=1.6)
    ax.axvline(C.T_Q, color="C0", ls=":", lw=1)
    ax.annotate("$T_Q$ = 48 K\n(noise-density knee)", (C.T_Q * 1.05, 3e-23),
                fontsize=8)
    ax.axvline(150, color="C2", ls=":", lw=1)
    ax.annotate("$\\tau$ saturates", (153, 3e-23), fontsize=8)
    ax.axvspan(4, 20, color="#eaf2fa")
    ax.annotate("error floors\n(vacuum-set)", (5, 1e-21), fontsize=8)
    ax.axvspan(300, 353, color="0.92")
    ax.annotate("Part-I band\n(300-353 K)", (302, 1e-21), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4, 400)
    ax.set_ylim(FLOOR, 1)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("error per operation")
    ax.set_title("Figure Q3. Error rate vs temperature: quantum-analog decode "
                 "against the classical fabric ($N_{op}$ = 400)")
    ax.legend(loc="center left", fontsize=8)
    ax.grid(alpha=0.25, which="both")
    ax.annotate("curves clipped at $10^{-24}$", (5, 2e-24), fontsize=7,
                style="italic")
    fig.savefig(os.path.join(OUT, "fig_q3.png"))
    plt.close(fig)


def fig_q4():
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0))
    sw = qerrors.sweep()

    A = ax[0]
    A.semilogx(sw["T"], sw["enob"], "k-")
    A.axvline(C.T_Q, color="C0", ls=":", lw=1)
    A.axhline(2, color="C3", ls=":", lw=1)
    A.annotate("2-bit decode depth", (5, 2.05), fontsize=8, color="C3")
    A.set_xlabel("temperature (K)")
    A.set_ylabel("per-shot analog precision (bits)")
    A.set_title("(A) effective bits of one MAC shot vs T")

    B = ax[1]
    Ns = np.logspace(1, 3.3, 40)
    for T, sty in ((300.0, "C3-"), (77.0, "C0-"), (4.0, "k-")):
        B.loglog(Ns, np.clip([qmac.error_2bit(T, N) for N in Ns], 1e-18, 1),
                 sty, label=f"2-bit, {T:.0f} K")
        B.loglog(Ns, np.clip([qmac.error_1bit(T, N) for N in Ns], 1e-18, 1),
                 sty, alpha=0.35, lw=1)
    B.axvline(400, color="0.5", ls=":", lw=1)
    B.annotate("$N_{op}$=400", (420, 0.3), fontsize=8)
    B.axvline(qmode.N_of_eps(0.01), color="C2", ls=":", lw=1)
    B.annotate("knee (preamp\nvariant cap)", (qmode.N_of_eps(0.01) * 1.1, 1e-14),
               fontsize=7.5, color="C2")
    B.set_xlabel("quanta per input $N_{op}$")
    B.set_ylabel("decode error")
    B.set_title("(B) error vs pulse quanta (faint: 1-bit)")
    B.legend(fontsize=8)
    fig.suptitle("Figure Q4. The analog precision budget", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_q4.png"))
    plt.close(fig)


def fig_q5():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 6.5)
    ax.axis("off")
    colors = {"guard": "0.85", "wave": "#dce9f5", "weight": "#fff3cd",
              "combine": "#fde2cf", "cell": "#d5e8d4", "tap": "#e8d5f0"}
    layout = {}
    for i in range(12):
        for j in range(6):
            layout[(i, j)] = "guard"
    for j in (1, 4):
        for i in range(0, 3):
            layout[(i, j)] = "wave"
    layout[(3, 1)] = "weight"; layout[(3, 4)] = "weight"
    for i in (4,):
        layout[(i, 1)] = "wave"; layout[(i, 4)] = "wave"
    layout[(5, 2)] = "combine"; layout[(5, 3)] = "combine"
    layout[(6, 2)] = "wave"; layout[(6, 3)] = "wave"
    layout[(7, 1)] = "cell"; layout[(7, 4)] = "cell"       # comparators
    layout[(8, 1)] = "wave"; layout[(8, 4)] = "wave"
    layout[(9, 1)] = "cell"; layout[(9, 4)] = "cell"       # NOT + buffer
    layout[(10, 2)] = "cell"                                # AND
    layout[(11, 2)] = "tap"; layout[(11, 4)] = "tap"        # read-out
    for (i, j), kind in layout.items():
        ax.add_patch(Rectangle((i, j), 0.94, 0.94, fc=colors[kind],
                               ec="w", lw=1.5))
    labels = [(0.5, 1.5, "B in"), (0.5, 4.5, "A in"),
              (3.5, 1.6, "$w$"), (3.5, 4.6, "$w$"),
              (5.95, 2.95, "combiner"), (7.5, 1.5, "$t_1$"), (7.5, 4.5, "$t_2$"),
              (9.5, 1.5, "NOT"), (9.5, 4.5, "BUF"), (10.5, 2.5, "AND"),
              (11.5, 2.7, "$b_0$"), (11.5, 4.7, "$b_1$")]
    for x, y, s in labels:
        ax.text(x, y - 0.25, s, ha="center", fontsize=8.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=c, label=l) for l, c in
                       [("CNP guard", "0.85"), ("waveguide", "#dce9f5"),
                        ("weight segment", "#fff3cd"),
                        ("junction combiner", "#fde2cf"),
                        ("regenerative cell", "#d5e8d4"),
                        ("read-out tap", "#e8d5f0")]],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=6,
              fontsize=7.5)
    ax.set_title("Figure Q5. QMAC-1 on the gate lattice: ~12 x 6 sites "
                 "(~13 x 14 um at the Part-I site pitch), five active cells - "
                 "the Part-I block, re-programmed")
    fig.savefig(os.path.join(OUT, "fig_q5.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_q1(); print("fig_q1.png")
    fig_q2(); print("fig_q2.png")
    fig_q3(); print("fig_q3.png")
    fig_q4(); print("fig_q4.png")
    fig_q5(); print("fig_q5.png")
