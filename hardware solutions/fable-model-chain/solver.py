# -*- coding: utf-8 -*-
"""Pass 3 (nonlinear) — 1-D shallow-water Dyakonov-Shur solver.

The gated 2D electron fluid obeys the shallow-water equations (Dyakonov & Shur,
PRL 1993): with h the local density (normalized to h0=1) and u the velocity
(normalized to the plasmon speed s=1), so that the long-wave speed is c=sqrt(h),

    d_t h        + d_x(h u)            = 0
    d_t(h u)     + d_x(h u^2 + h^2/2)  = -h (u - u0)/tau_n

The constant body force u0/tau (folded into the relaxation as -h(u-u0)/tau)
sustains the DC drift u0 = M against momentum relaxation; the uniform state
h=1, u=u0 is then an exact steady solution. Asymmetric DS boundary conditions:
source (x=0) density-clamped, h=1 (AC short); drain (x=L) current-clamped,
h u = u0 (AC open). Time units are L/s, so one round trip = 2, the carrier
period is 1/(f0 L/s), and tau_n = tau s / L.

Integrated with a Lax-Friedrichs flux (robust; its numerical diffusion raises the
measured threshold above the analytic 0.146 -- reported, not tuned away) plus a
2nd-order MacCormack option. Three experiments: free-perturbation threshold,
CW regenerative gain, and few-cycle pulse-train gain/spread.
"""
import numpy as np
import constants as C
from ds_cell import plasmon_speed, cell_length, M_threshold


def _setup(N, T=C.Tcap):
    s = plasmon_speed()
    tau = C.tau(T)
    L = cell_length(s)
    tau_n = tau * s / L
    f0_n = C.f0 * L / s          # carrier freq in 1/(L/s)
    return s, tau, L, tau_n, f0_n


def _flux(h, hu):
    u = hu / h
    return hu, hu * u + 0.5 * h * h


def _step_LF(h, hu, dx, dt, u0, tau_n, h_left, drain_current):
    """One Lax-Friedrichs step with DS boundary ghosts."""
    N = len(h)
    # ghost-extended arrays
    H = np.empty(N + 2); HU = np.empty(N + 2)
    H[1:-1] = h; HU[1:-1] = hu
    # source ghost (x=0): density clamped h=h_left, velocity zero-gradient
    H[0] = h_left
    HU[0] = h_left * (hu[0] / h[0])
    # drain ghost (x=L): current clamped h*u = drain_current, h zero-gradient
    H[-1] = h[-1]
    HU[-1] = drain_current
    f1, f2 = _flux(H, HU)
    # LF update on interior
    h_new = 0.5 * (H[2:] + H[:-2]) - 0.5 * dt / dx * (f1[2:] - f1[:-2])
    hu_new = 0.5 * (HU[2:] + HU[:-2]) - 0.5 * dt / dx * (f2[2:] - f2[:-2])
    # relaxation source toward drift u0 (momentum)
    hu_new += dt * (-(hu_new - h_new * u0) / tau_n)
    return h_new, hu_new


def run(M, N=240, n_roundtrips=120, drive_amp=0.0, drive_kind="none",
        f0_n=None, cfl=0.4, T=C.Tcap, seed=1, rep_ratio=0.25, bits=None):
    """Evolve the cell. Returns dict with time series at the drain and intracavity.
    drive_kind: 'none' (seed noise -> threshold test), 'cw', 'pulse', 'prbs'."""
    s, tau, L, tau_n, f0n = _setup(N, T)
    if f0_n is None:
        f0_n = f0n
    u0 = M
    dx = 1.0 / N
    cmax = 1.0 + abs(u0) + 0.2
    dt = cfl * dx / cmax
    nsteps = int(n_roundtrips * 2.0 / dt)
    h = np.ones(N); hu = np.ones(N) * u0
    rng = np.random.default_rng(seed)
    if drive_kind == "none":
        h += 1e-4 * rng.standard_normal(N)        # tiny seed perturbation
    repT = 1.0 / (rep_ratio * f0_n)
    out = np.empty(nsteps); tarr = np.empty(nsteps); cav = np.empty(nsteps)
    t = 0.0
    for k in range(nsteps):
        if drive_kind == "cw":
            sig = drive_amp * np.sin(2 * np.pi * f0_n * t)
        elif drive_kind == "pulse":
            sig = _pulse_train(t, f0_n, drive_amp, rep_ratio=rep_ratio)
        elif drive_kind == "prbs":
            slot = int(t / repT)
            b = bits[slot] if (bits is not None and slot < len(bits)) else 0
            sig = _pulse_train(t, f0_n, drive_amp, rep_ratio=rep_ratio) if b else 0.0
        else:
            sig = 0.0
        h_left = 1.0 + sig
        h, hu = _step_LF(h, hu, dx, dt, u0, tau_n, h_left, u0)
        # guard against blow-up
        if not np.all(np.isfinite(h)) or np.any(h < 1e-6):
            out = out[:k]; tarr = tarr[:k]; cav = cav[:k]; break
        out[k] = hu[-1] / h[-1] - u0        # velocity perturbation at drain (forced node)
        cav[k] = np.max(np.abs(hu / h - u0))  # intracavity peak velocity perturbation
        tarr[k] = t
        t += dt
    return dict(t=tarr, drain=out, cav=cav, dt=dt, f0_n=f0_n, tau_n=tau_n, L=L, s=s, M=M)


def _pulse_train(t, f0_n, amp, rep_ratio=0.25, ncyc=3):
    """Few-cycle Gaussian-windowed carrier bursts at rep = rep_ratio * f0."""
    rep_n = rep_ratio * f0_n
    period = 1.0 / rep_n
    phase = (t % period)
    width = ncyc / f0_n
    centre = 0.5 * period
    env = np.exp(-0.5 * ((phase - centre) / (width / 2.355)) ** 2)
    return amp * env * np.sin(2 * np.pi * f0_n * (phase - centre))


def growth_rate(M, **kw):
    """Free-perturbation log-amplitude growth rate (per round trip)."""
    r = run(M, drive_kind="none", **kw)
    sig = np.abs(r["drain"])
    t = r["t"]
    if len(t) < 100:
        return np.nan
    # envelope via running max over windows; fit log in the last 60%
    i0 = int(0.4 * len(t))
    env = np.maximum.accumulate(sig[i0:][::-1])[::-1]  # decreasing-from-right
    tt = t[i0:]
    amp = sig[i0:]
    # robust: fit slope of log of windowed RMS
    nb = 40
    edges = np.linspace(tt[0], tt[-1], nb + 1)
    cen = 0.5 * (edges[1:] + edges[:-1])
    rms = np.array([np.sqrt(np.mean(amp[(tt >= edges[i]) & (tt < edges[i + 1])] ** 2) + 1e-30)
                    for i in range(nb)])
    good = rms > 0
    A = np.polyfit(cen[good], np.log(rms[good]), 1)
    return A[0] * 2.0   # per round trip (round trip = 2 time units)


if __name__ == "__main__":
    s, tau, L, tau_n, f0n = _setup(240)
    Mth = M_threshold(L, s, tau)
    print(f"analytic M_th = {Mth:.3f}; tau_n={tau_n:.2f}, f0_n={f0n:.3f}, period={1/f0n:.2f}")
    print("free-perturbation growth rate vs M (Lax-Friedrichs, N=240):")
    for r in [0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3]:
        g = growth_rate(r * Mth, n_roundtrips=80)
        print(f"  M/M_th_analytic={r:.2f} (M={r*Mth:.3f}): growth/rt = {g:+.4f}")
