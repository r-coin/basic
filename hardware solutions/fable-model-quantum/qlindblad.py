# -*- coding: utf-8 -*-
"""Fock-space verification of the Gaussian rules used by qnoise.

The Gaussian calculus is only as good as its two elementary rules. Both are
checked here against exact (truncated-Fock-space) quantum evolution:

  1. LOSS + THERMAL BATH. The Lindblad master equation
         drho/dt = kappa (nbar+1) D[a] rho + kappa nbar D[a'] rho
     evolved from a coherent state must reproduce
         <x>(t) = sqrt(2N) e^(-kappa t / 2),
         V(t)   = 1/2 + nbar (1 - e^(-kappa t)),
     i.e. the qnoise loss rule with eta = e^(-kappa t).

  2. PHASE-INSENSITIVE AMPLIFIER. The two-mode squeeze
         U = exp[ r (a' b' - a b) ],   G = cosh^2 r,
     acting on |alpha> (signal) x (idler at nbar_i), then tracing the idler,
     must reproduce
         <x> -> sqrt(G) <x>,   V -> G V + (G-1)(nbar_i + 1/2),
     the qnoise amp rule (Caves limit at nbar_i = 0; the parent's Eq. 7
     matched-temperature floor at nbar_i = nbar(T)).

Both integrations use RK4 with generous cutoffs; the printed relative errors
are truncation-level (<1%). No external packages beyond numpy.
"""
import math
import numpy as np
import qconstants as C


def _ops(dim):
    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    return a, a.conj().T


def _coherent_np(alpha, dim):
    n = np.arange(dim)
    lg = np.array([math.lgamma(k + 1) for k in n])
    amp = np.exp(-0.5 * alpha**2 + n * math.log(alpha) - 0.5 * lg) \
        if alpha > 0 else np.eye(dim)[0]
    return amp / np.linalg.norm(amp)


def check_loss_thermal(alpha=3.0, nbar=2.0, eta_target=0.5, dim=64, nsteps=400):
    """Verify the loss rule against the exact Lindblad solve."""
    a, ad = _ops(dim)
    n_op = ad @ a
    x_op = (a + ad) / math.sqrt(2)
    psi = _coherent_np(alpha, dim)
    rho = np.outer(psi, psi.conj())

    kappa = 1.0
    t_end = -math.log(eta_target) / kappa
    dt = t_end / nsteps

    L1 = math.sqrt(kappa * (nbar + 1)) * a
    L2 = math.sqrt(kappa * nbar) * ad

    def lind(r):
        out = np.zeros_like(r)
        for Lk in (L1, L2):
            Ld = Lk.conj().T
            out += Lk @ r @ Ld - 0.5 * (Ld @ Lk @ r + r @ Ld @ Lk)
        return out

    for _ in range(nsteps):
        k1 = lind(rho)
        k2 = lind(rho + 0.5 * dt * k1)
        k3 = lind(rho + 0.5 * dt * k2)
        k4 = lind(rho + dt * k3)
        rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    x_num = float(np.real(np.trace(rho @ x_op)))
    x2 = float(np.real(np.trace(rho @ x_op @ x_op)))
    V_num = x2 - x_num**2
    x_th = math.sqrt(2) * alpha * math.sqrt(eta_target)
    V_th = eta_target * 0.5 + (1 - eta_target) * (nbar + 0.5)
    return (x_num, x_th, V_num, V_th)


def check_amplifier(alpha=2.0, G=2.0, nbar_i=0.0, dim_a=96, dim_b=72,
                    nsteps=400, m_max=16):
    """Verify the amplifier rule against the exact two-mode squeeze.

    For nbar_i > 0 the idler thermal state is handled as a weighted mixture of
    Fock states (geometric weights), each evolved as a pure state. The
    generator is applied matrix-free on the (dim_a, dim_b)-reshaped state, so
    large cutoffs stay cheap.
    """
    r = math.acosh(math.sqrt(G))
    a, ad = _ops(dim_a)
    b, bd = _ops(dim_b)
    xa = (a + ad) / math.sqrt(2)

    psi_a = _coherent_np(alpha, dim_a).astype(complex)

    # idler Fock weights
    if nbar_i == 0:
        weights = [(0, 1.0)]
    else:
        p = nbar_i / (nbar_i + 1)
        weights = [(m, (1 - p) * p**m) for m in range(m_max)]

    def K_apply(P):
        # (ad x bd - a x b) on the reshaped state: ad P bd^T - a P b^T
        return ad @ P @ bd.T - a @ P @ b.T

    ds = r / nsteps
    xs, x2s, wtot = 0.0, 0.0, 0.0
    for m, wgt in weights:
        P = np.zeros((dim_a, dim_b), dtype=complex)
        P[:, m] = psi_a
        for _ in range(nsteps):
            k1 = K_apply(P)
            k2 = K_apply(P + 0.5 * ds * k1)
            k3 = K_apply(P + 0.5 * ds * k2)
            k4 = K_apply(P + ds * k3)
            P = P + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        P /= np.linalg.norm(P)
        rho_a = P @ P.conj().T                    # signal reduced density matrix
        xs += wgt * float(np.real(np.trace(rho_a @ xa)))
        x2s += wgt * float(np.real(np.trace(rho_a @ xa @ xa)))
        wtot += wgt
    xs, x2s = xs / wtot, x2s / wtot
    V_num = x2s - xs**2
    x_th = math.sqrt(2) * alpha * math.sqrt(G)
    V_th = G * 0.5 + (G - 1) * (nbar_i + 0.5)
    return (xs, x_th, V_num, V_th)


if __name__ == "__main__":
    print("1) loss + thermal bath (alpha=3, nbar=2, eta=0.5):")
    xn, xt, Vn, Vt = check_loss_thermal()
    print(f"   <x>: exact {xn:.4f} vs rule {xt:.4f}  "
          f"({abs(xn-xt)/xt*100:.3f} % off)")
    print(f"   V  : exact {Vn:.4f} vs rule {Vt:.4f}  "
          f"({abs(Vn-Vt)/Vt*100:.3f} % off)")

    print("2) amplifier, vacuum idler (alpha=2, G=2 — Caves limit):")
    xn, xt, Vn, Vt = check_amplifier()
    print(f"   <x>: exact {xn:.4f} vs rule {xt:.4f}  "
          f"({abs(xn-xt)/xt*100:.3f} % off)")
    print(f"   V  : exact {Vn:.4f} vs rule {Vt:.4f}  "
          f"({abs(Vn-Vt)/Vt*100:.3f} % off)")

    print("3) amplifier, thermal idler nbar=1 (matched-T floor, Eq. 7):")
    xn, xt, Vn, Vt = check_amplifier(nbar_i=1.0)
    print(f"   <x>: exact {xn:.4f} vs rule {xt:.4f}  "
          f"({abs(xn-xt)/xt*100:.3f} % off)")
    print(f"   V  : exact {Vn:.4f} vs rule {Vt:.4f}  "
          f"({abs(Vn-Vt)/Vt*100:.3f} % off)")
