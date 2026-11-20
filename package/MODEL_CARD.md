# Model card: Merger Transformer v0.5.0

## Identity and intended use

This release distributes the extended-window width-224 Merger Transformer selected at optimizer update `64,000` (`float64` EMA weights, `2,564,276` parameters). It corresponds to Jameson Dong, *A high-precision Transformer model for precessing binary black-hole merger waveforms calibrated to numerical relativity* (November 20, 2026).

The package provides standalone inference, automatic parameter and time differentiation, observer projection, and optional local private-NR fine-tuning. Training scripts, NR catalogues, synthetic teacher caches, and optimizer states are not distributed.

## Interface and conventions

| Quantity | Contract |
| --- | --- |
| Intrinsic input | `(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)` |
| Mass ratio | `q = m1/m2 > 0`; smooth across `q = 1` without sorting |
| Spins | Dimensionless Cartesian vectors in the reference triad, `|chi_a| <= 1` |
| Reference frequency | `M f_ref = 0.01182117827433904` (20 Hz at `120 M_sun`) |
| Reference axes | `(n, Lhat × n, Lhat)` at `M f_ref`; `n` points from body 2 to body 1 |
| Native time domain | `tau = t/M` over closed support `[-1620, 160]M` |
| Evaluation window | Primary `[-1500, 100]M` (`[-1000, 40]M` noted for reference) |
| Strain output | Distance-scaled modes `H_lm = (R/M) h_lm`, `h = h_plus - i h_cross` |
| Modes | All 21 signed spin-weight `-2` modes for `ell = 2,3,4` (`m = -ell,...,ell`) |
| Curvature output | `P_lm = -d²H_lm/dtau²` analytically from the same spline coefficients |
| Reference precision | `float64` / `complex128` on CPU and CUDA (`float32` option / MPS) |

## Architecture and spline basis

The shared coefficient network (`2,564,276` parameters) predicts `512` complex quintic B-spline coefficients per mode:

- Input scaling `(log(q)/log(4) - 0.5, chi1, chi2)` passed through a three-layer width-256 SiLU MLP and linearly projected to `8` width-224 conditioning memory tokens.
- Three sequential Transformer decoder blocks (`width = 224`, `4` heads, pre-LayerNorm cross-attention, FFN hidden width `448` with GELU, and pre-LayerNorm self-attention over all `512` spline tokens), plus linear readout (`224 -> 42`).
- Additive mode-attention branch projecting each normalized spline token to `21` width-`32` mode tokens with learned mode embeddings, two `4`-head self-attention/FFN blocks (`FFN width 64`), zero-initialized per-mode readouts, and fixed `0.3` readout scaling.
- Exact four-element Reynolds average over `{I, E, P, EP}`, multiplied by `s(nu) = [1 + (0.05/nu)^4]^(-1/4)` where `nu = q/(1+q)^2`.

The degree-5 B-spline basis has `518` knots: multiplicity-6 clamped endpoints at `-1620M` and `160M`, and `506` simple interior breaks spaced uniformly in the cumulative integral of the two-sided analytic bump density:

```text
rho(tau) = 1 + 3 * [sigma((tau + 200)/40) - sigma((tau - 75)/15)],   sigma(z) = 1 / (1 + e^-z)
```

Across interior breaks, `h_lm(tau; x)` is `C^4` and `Psi_{4,lm}(tau; x)` is `C^2`, while `c(x)` is real analytic on `q > 0` for fixed weights.

## Training provenance and selection

Direct NR partition: **2,697 training waveforms (2,347 physical-system groups), 166 validation waveforms (150 groups), and 282 evaluation waveforms (266 groups)**. Upstream teacher fitting/parent families are assigned to training/validation. Auxiliary training regressions—reference-epoch calibration maps `Delta_hat` (`798/783`) and `d_hat` (`1282/1220`), plus a time-independent, parameter-dependent active `SO(3)` fit on `43/43` SXS precessing simulations (`q in [5,6]`) for the precessing `SEOBNRv5PHM` teachers—have zero validation or evaluation overlap.

Joint pretraining (updates `0–32,000`) combines the NR sample with **41,444 synthetic waveforms** (`33,000` `NRSur7dq4 v1`, `8,144` `pyseobnr 0.3.6` aligned and precessing `SEOBNRv5PHM` with antisymmetric modes up to `q = 20`, and `300` aligned `BHPTNRSur2dq1e3` up to `q = 1000`), annealing synthetic and RIT weights to zero by update `32,000`. Subsequent SXS-only fine-tuning runs on `2,038` SXS training waveforms with late-ringdown `D2` weight `0.02` (`32,000–52,000`) and `0.40` (`52,000–64,000`) at learning rate `1e-5` with an unaligned core anchor (`mu = 0.01`). Validation selection over `165` waveforms (`149` groups, excluding `SXS:BBH:2633`) achieves an SXS validation group-mean strain error of `0.0285` at the selected `STEP064000` `float64` EMA checkpoint.

## Evaluation performance (`[-1500, 100]M`)

All metrics below are physical-system group means on `[-1500, 100]M` (with `120 M_sun` Advanced LIGO zero-detuned high-power mismatch):

- **Full SXS evaluation panel (`109` waveforms / `106` groups)**: `L2(h) = 0.0305`, `L2(Psi4) = 0.1497`, mean mismatch `0.1536%`, P95 `0.7870%`. (On `[-1000, 40]M` core reference: `L2(h) = 0.0291`, `M(120) = 0.1470%`.)
- **Common SXS support (`105` waveforms / `102` groups)**:
  - Transformer: `L2(h) = 0.0301`, `L2(Psi4) = 0.1487`, mean mismatch `0.1496%`, P95 `0.7446%`.
  - `NRSur7dq4v2`: `0.0679`, `0.2155`, `0.6049%`, `3.2233%`.
  - `SEOBNRv5PHM`: `0.1124`, `0.3262`, `1.5475%`, `11.3127%`.
  - `IMRPhenomTPHM`: `0.1546`, `0.3752`, `2.5983%`, `19.3484%`.
- **`NRSur7dq4v2` calibration domain (`69` SXS waveforms / `68` groups, `q <= 4.01`, `chi <= 0.8`)**: Transformer `L2(h) = 0.0184` vs `0.0240` (`paired diff -0.0056, 95% CI [-0.0112, -0.0011]`), mean mismatch `0.0253%` (P95 `0.0549%`) vs `0.0326%` (P95 `0.0831%`).
- **RIT evaluation panel (`173` waveforms / `163` groups)**: `L2(h) = 0.0809` overall (`0.0525` aligned, `0.1176` precessing); `0.0741` vs `NRSur7dq4v2` `0.0721` on common support (`167/157`).
- **Auxiliary native BAM curvature (`R=90M`, `[-160, 40]M`)**: relative `L2(Psi4)` of `0.087, 0.104` (`q ~ 1, 2`), `0.200, 0.222` (`q ~ 4`), and `0.423, 0.436` (`q ~ 8`).
- **Peak-aligned reference diagnostics (`[-160, 20]M`)**: fixed-azimuth `0.2414`, single rigid `Rz` `0.1338`, median `|Delta alpha| = 3.16 deg`, median `|Delta t_peak| = 0.66M`.

## Physical scope and limitations

- **Duration**: waveforms span `tau in [-1620, 160]M` (`[-1500, 100]M` evaluated). Low-mass full-band analyses entering detector band earlier than `-1500M` require hybridization with an earlier inspiral.
- **High-mass-ratio precession**: public SXS precessing evaluation is concentrated at `q <= 6` (`24` groups in `4 < q <= 6`, `1` in `6 < q <= 8`, `0` at `q > 8`); predictions at higher `q` rely on synthetic pretraining.
- **Reference frame**: aligned comparisons profile `(delta, alpha)`; applications requiring strict unaligned phase accuracy must account for residual time and azimuth offsets.

## License and credit

Code, documentation, and weights are licensed under MIT ([LICENSE](LICENSE)). Scientific use should cite the paper and upstream NR/approximant sources (SXS, RIT, Cardiff–BAM, `NRSur7dq4`, `SEOBNRv5PHM`, `BHPTNRSur2dq1e3`).
