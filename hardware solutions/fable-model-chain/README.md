# Fable Computer — reduced-order model chain

Runnable reproduction of the five-pass feasibility chain and the energy/thermal
accounting behind Sections 7–8 of *The Fable Computer* (room-temperature THz
half adder on a regenerative graphene-plasmon fabric). Pure Python; depends only
on `numpy` and (for figures) `matplotlib`.

## Quick start

```bash
python3 run_all.py --json      # prints every manuscript number; writes results.json
python3 figures.py             # regenerates Figures 6–10 into figures/
```

Each module is also runnable on its own and prints a self-check against the
manuscript values (e.g. `python3 ds_cell.py`, `python3 kinetic.py`).

## Modules

| File | Pass / role | Key outputs |
|------|-------------|-------------|
| `constants.py` | operating point, materials | E_F, Q, τ(T), v_sat, R_th |
| `ds_cell.py`   | Eqs (1)–(4), Pass 2 | plasmon speed, M_th, per-gate loss |
| `solver.py`    | Pass 3 (nonlinear) | 1-D shallow-water DS solver; M_th,num; saturation; pulse train |
| `regen.py`     | Pass 3 (analytic) | regenerative cavity gain (CW + pulse), threshold |
| `cell.py`      | Eq (5), logic | transfer model, compression, cascade, noise margin, ISI, truth table |
| `noise.py`     | Pass 4 | F = 2 − 1/G, re-sync spacing |
| `disorder.py`  | Pass 4 | Monte-Carlo cell yield vs charge-puddle σ |
| `thermal.py`   | Pass 1 + §8 | Joule heat, ΔT vs fill, duty ceilings, energy/add |
| `kinetic.py`   | Pass 5 | τ_ee, ωτ_ee, viscous fraction, validity temperature |
| `coupled.py`   | Pass 5 | coupled two-cell stability vs midpoint clamp |
| `run_all.py`   | driver | consolidated table + `results.json` |
| `figures.py`   | figures | Figures 6–10 |

## Method notes (what each pass actually computes)

- **Analytic backbone** (Eqs 1–7, thermal, kinetic) is closed-form and
  reproduces the operating point from first principles: E_F = 117 meV,
  Q = 6.3/5.3, M_th = 0.125/0.147, per-gate loss 2.17/2.56 dB, τ_ee = 115 fs.
- **Pass 3 gain** is owned by the analytic regenerative-cavity transfer in
  `regen.py` (Fabry-Perot below threshold: loop = (s+v₀)/(s−v₀)·a_loss, gain =
  [(1−loop₀)/(1−loop)]²). The **nonlinear shallow-water solver** (`solver.py`)
  independently confirms the sub-threshold regenerative buildup, locates the
  numerically-diffused threshold M_th,num ≈ 0.17 (≈15 % above analytic; a
  Lax-Friedrichs diffusion artifact that → analytic as Δx→0), and supplies the
  gain compression (1 dB at ≈1 % swing) and the streaming per-pulse gain.
- **Per-pulse gain.** A 3-cycle burst is shorter than the low-Q (Q≈6) cavity
  buildup time, so its gain (≈ +8 dB streaming, solver) sits below the CW
  regenerative gain (+9.7 dB). The cavity ring-down (~4–5 ps) is comparable to
  the 4 ps slot, so data-pattern ISI is controlled by the architecture's
  per-slot bias gating / re-sync flushing, not assumed away.
- **Yield criterion** (`disorder.py`) ties cell yield to the k≥8 sharpness rule:
  a puddle of rms σ broadens the threshold to √((n/k₀)²+σ²); the cell yields if
  the broadened sharpness clears the regeneration floor.

## Reproducibility caveat

These are deliberately transparent *reduced-order* models, not a Boltzmann–
Maxwell solve. The numerically-diffused threshold offset, the absolute pulse-gain
calibration, and the yield criterion's free parameters are documented in-line.
The remaining tier (full kinetic + electromagnetic) is Section 10's open work.
