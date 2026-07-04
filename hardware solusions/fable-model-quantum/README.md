# Fable Computer, Part II — quantum extension model chain

Runnable companion to *The Fable Computer, Part II: Quantum-Limited Analog
Tensor Processing on the Graphene-Plasmon Fabric*. It quantizes the Part-I
DS cell mode, builds the minimal quantum-analog tensor unit (QMAC-1: two
inputs, one interference, one 2-bit classically-decoded output), and computes
error rates vs temperature against the classical digital fabric. Pure Python;
depends only on `numpy` and (for figures) `matplotlib`.

## Quick start

```bash
python run_all.py --json      # prints every Part-II number; writes results.json
python figures.py             # regenerates Figures Q1-Q5 into figures/
```

Each module is independently runnable and prints a self-check
(e.g. `python qmode.py`, `python qlindblad.py`).

## Modules

| File | Role | Key outputs |
|------|------|-------------|
| `qconstants.py` | operating point + quantum scales | hbar*w0, T_Q = 48 K, nbar(T), tau_q(T) with 150-K saturation; parent anchors reproduced |
| `qmode.py`      | DS-mode quantization | eps_1 = 1.6e-3 per plasmon, knee = 38 quanta, rail = 3.8e3; Kerr chi/kappa ~ 1e-5 (the gate-model no-go), 3-nm blockade cell |
| `qnoise.py`     | Gaussian quadrature calculus | loss/amp/combine rules, matched-T amplifier = parent Eq. (7), symbol error + MC check |
| `qdecode.py`    | classical 2-bit flash decode | 2 comparators + NOT + AND + buffer = the Part-I half-adder block; decoder noise; truth table b1b0 = A+B |
| `qmac.py`       | QMAC-1, the tensor unit | levels at decision, 2-bit/1-bit error, preamp-variant no-go, weighted-MAC demo, ENOB |
| `qlindblad.py`  | Fock-space verification | loss+thermal Lindblad and two-mode-squeeze amplifier reproduce the Gaussian rules to <0.01 % |
| `qerrors.py`    | temperature sweeps | error vs T for all machines; threshold temperatures; quantum floors |
| `run_all.py`    | driver | consolidated table + `results.json` |
| `figures.py`    | figures | Figures Q1-Q5 |

## Headline results (N_op = 400 quanta/input unless noted)

- Logic pulses are mesoscopic: the Part-I 1-dB knee is **38 plasmons**, the
  rail **3.8e3** — quantum noise is quantitative here, not academic.
- Gate-model (blockade) quantum logic misses by **five orders of magnitude**
  in this cell (chi/kappa ~ 1e-5; needs ~3-nm cells) — the extension is
  quantum-LIMITED analog computing, stated as such.
- The analog core must stay **passive** (linear interference); a regenerative
  preamp caps the linear window at the 38-quantum knee and loses at every
  temperature. Gain belongs after the decision, in the decoder cells.
- **2-bit decode**: 0.14 error at 300 K (unusable per-shot), 3.1e-3 at 77 K,
  quantum floor 1.3e-6 below ~15-20 K (1.3e-10 at N_op = 800).
- **1-bit (full-range binary) decode**: 3.9e-3 at 300 K, 4.8e-9 at 77 K —
  "lower-bit decoding" works two temperature classes earlier than 2-bit.
- **16-slot averaging** (charge-memory accumulation; per-slot noise averages,
  the static threshold band does not) buys the 2-bit decode to ~5e-7 at
  300 K at 1/16th throughput (still 1.6e10 MAC/s).
- The **classical digital fabric** (rail-restored, ~3.8e3 quanta/decision;
  this BER model is Part II's own construction — Part I quotes no BER)
  stays below ~1e-9 in the Part-I band — digital restoration beats the
  analog unit at 300 K by restoring at ~10x the analog unit's per-input
  amplitude and giving up linearity. The quantized noise floor makes
  rail-scale restoration a requirement: a knee-level (38-quantum) digital
  swing would be thermal-limited to ~1e-1 BER at 300 K.
- A **20-dB-SNR classical launch** (fixed fraction of the full swing) pins
  every decode near 1e-2 at all temperatures: quantum operation requires a
  near-shot-noise launch — the sharpest new hardware requirement of Part II.
- **T_Q = 48 K** is the noise-density knee (thermal -> vacuum dominated);
  the error curves, being erfc tails, saturate at their vacuum-set floors
  below ~15-20 K. That saturation is the quantum limit, and its observation
  is a pre-registered bench gate.
- **Junction sensitivity**: the default assumes the optimistic end (-1 dB)
  of the parent's -1 to -3 dB junction band. At -3 dB junctions the 2-bit
  error degrades to 5.6e-2 at 77 K (from 3.1e-3) and the 1-bit error to
  1.8e-4 at 77 K (from 4.8e-9): junction quality is a first-order lever on
  every cold-class number, and the -3 dB case is reported alongside the
  default in results.json (junction_sensitivity_3dB).

## Reproducibility caveat

Reduced-order, deliberately transparent models in the Part-I tradition: exact
Gaussian propagation (verified against Fock-space quantum evolution in
`qlindblad.py`) over a stagewise loss/gain budget taken from Part I. The Kerr
coefficient carries an order-of-magnitude band; launch coupling efficiency and
absolute noise calibration remain open items owned by the Boltzmann-Maxwell
tier and the bench, exactly as in Part I.
