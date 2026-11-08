# Model card: Merger Transformer v7

## Identity and intended use

This package provides the v7 width-192 recovery Transformer selected at optimizer update 48,000 after constant-low-learning-rate continuation. Inference uses its float64 exponential-moving-average (EMA) parameters. It corresponds to Jameson Dong, *A Transformer model for precessing binary black-hole merger waveforms calibrated to numerical relativity*, v7 manuscript.

The intended use is waveform research: generating finite-window, multimode binary-black-hole merger signals and differentiating the same model with respect to intrinsic parameters and time. The package provides standalone inference and final weights. It does not contain a training pipeline, numerical-relativity targets, synthetic teacher caches, or the original optimizer state.

## Interface and conventions

| Quantity | Contract |
| --- | --- |
| Intrinsic input | `(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)` |
| Mass ratio | `q = m1/m2 > 0`; no sorting or cutoff |
| Spins | Dimensionless Cartesian vectors, each with magnitude at most one |
| Reference frequency | `M f_ref = 0.01182117827433904`, equivalent to 20 Hz at 120 solar masses |
| Reference axes | `(n, Lhat × n, Lhat)` at that epoch; `n` points from body 2 to body 1 |
| Native time | `tau = t/M` in geometrized units; closed support `[-1100, 80]` |
| Strain output | Distance-scaled modes `H_lm = (R/M) h_lm`, with `h = h_plus - i h_cross` |
| Modes | All 21 signed spin-weight `-2` modes, ordered by ascending `ell = 2,3,4`, then `m = -ell,...,ell` |
| Curvature output | `P_lm = -d²H_lm/dtau²`, derived analytically from the same strain coefficients |
| Default numerical precision | Real float64, complex128 output |

Mass and distance scaling, observer projection, and detector response are subsequent operations. [README.md](README.md) gives explicit physical-unit conversions and API examples. The interface does not transport spins to the reference epoch, align the output to a supplied waveform, or recenter its peak.

The NR target clock is a weighted time moment of total curvature power, distinct from the spin-reference event and the strain-amplitude peak. A weak unaligned SXS strain anchor and fixed-azimuth in-domain surrogate supervision constrain the learned time and azimuth during training. There is no adopted inference-time correction to those quantities.

## Architecture and inference payload

The network has **1,974,036 trainable parameters** and predicts 384 complex quintic B-spline coefficients for each mode. The coefficient map uses:

- A three-layer width-256 SiLU conditioner on `(log(q)/log(4) - 0.5, chi1, chi2)`, projected to eight width-192 memory tokens.
- Three coefficient-decoder blocks combining four-head cross-attention, a width-384 GELU feed-forward network, and self-attention over 384 coefficient tokens.
- An additive mode-attention head of width 32, with two four-head blocks over the 21 modes at each coefficient index.
- A shared four-evaluation exchange/reflection average, followed by the fixed factor `s(nu) = [1 + (0.05/nu)**4]**(-1/4)`, where `nu = q/(1+q)**2`.

The spline has 390 knots: six copies of each endpoint and 378 simple interior knots. Interior breaks are uniformly spaced in the cumulative integral of the density `rho(tau) = 1 + 3 / [1 + exp(-(tau + 200)/40)]`. The payload preserves the exact stored float64 knot values. The internal amplitude factor `0.3` and the symmetric-mass-ratio factor are part of inference, not scaling choices left to the caller.

The exported inference state consists of the 104 EMA parameter tensors, the fixed knot buffer, and the configuration needed to interpret them. Raw optimization parameters are not used for inference. No optimizer, scheduler, random-number-generator, sampler, or alignment-cache state is needed to generate waveforms.

## Symmetry and differentiability

For `R = diag(-1,-1,1)`, body exchange and equatorial reflection act on inputs as

```text
E(q, chi1, chi2) = (1/q, R chi2, R chi1)
P(q, chi1, chi2) = (q, R chi1, R chi2).
```

Their mode actions are `E H_lm = (-1)**m H_lm` and `P H_lm = (-1)**ell conjugate(H_l,-m)`. Exchanging the bodies therefore requires both the spin-slot swap and the two in-plane sign changes. The network does not canonicalize `q` to values above one. The same relations apply to curvature and coefficients.

The architecture enforces these relations in exact arithmetic; numerical evaluations agree to floating-point precision. Aligned-spin inputs obey `H_l,-m = (-1)**ell conjugate(H_lm)`. Odd-`m` modes vanish for exchange-invariant configurations, not for every generic precessing system with equal masses.

With fixed weights, the coefficient function is real analytic on `q > 0` in exact arithmetic. The quintic basis makes strain `C4` and curvature `C2` across simple interior time knots. Endpoint derivatives are the one-sided spline derivatives. These are properties of the representation: they do not establish agreement with the physical waveform or its derivatives, and they do not imply unlimited-order time smoothness. PyTorch input/time autograd is preserved while the inference weights remain frozen.

## Training provenance and selection

The direct NR split contains **2697 training waveforms (2347 physical-system groups), 166 validation waveforms (150 groups), and 282 evaluation waveforms (266 groups)**. SXS and RIT are separated in accuracy reporting. SXS input spins and the reference triad come from apparent-horizon trajectories. SXS strain targets subtract the catalogue-added BMS energy-flux memory contribution. RIT reference labels use the inherited coupled quasicircular SEOB-based transport with the adopted waveform-clock/radiative-frame repairs; aligned-spin targets use published strain and precessing targets use reconstructed strain from Psi4, with the documented drift removal. The two catalogues have different preparation uncertainties. No subsequently developed RIT relabeling or time/azimuth correction is applied to this checkpoint.

Joint training uses **41,444 synthetic waveforms**:

| Teacher population | Count | Sampling scope |
| --- | ---: | --- |
| NRSur7dq4 v1, in domain | 24,000 | `1 <= q <= 4`, spin magnitudes <= 0.8 |
| NRSur7dq4 v1, q extrapolation | 8,000 | `4 < q <= 6`, spin magnitudes <= 0.8 |
| NRSur7dq4 v1, aligned high spin | 1,000 | `1 <= q <= 4`, maximum spin magnitude 0.8–0.95 |
| SEOBNRv5PHM, aligned | 2,000 | `6 <= q <= 20`, aligned spins -0.8–0.8 |
| SEOBNRv5PHM, precessing | 6,144 | 4,608 at q8–20 plus 1,536 at q6–8; spin magnitudes up to 0.9 |
| BHPTNRSur2dq1e3, aligned | 300 | `20 < q <= 1000`, primary spin -0.5–0.8, secondary spin zero |

The SEOB teacher uses stock pyseobnr 0.3.6 with antisymmetric (2,2), (3,3), and (4,4) corrections. Its precessing targets supervise a rank-12 rotating mode subspace. Unmodeled teacher modes are masked. NRSur7dq4 **v1** is the teacher; **v2** is only an evaluation comparator. Reference-time adapters reconcile the teacher and horizon-reference conventions. Synthetic coverage does not establish NR calibration throughout those parameter ranges.

Training uses strain, finite-difference and tail terms without an explicit analytic Psi4 objective. Synthetic and RIT weights anneal to zero by update 32,000; SXS-only fine-tuning uses 2038 training waveforms and a soft unaligned strain anchor. Adam state and EMA first continue to update 40,000, then the selected constant-learning-rate arm continues the exact full state for 8,000 more updates at `1e-5`, with no reheating. The user selected the update-48,000 EMA as the new product. Its external SXS validation group-mean strain/curvature errors are **0.03388809 / 0.14252410** on 82 waveforms in 78 groups. Checkpoint-embedded validation fields are inherited from update 40,000 and are not the endpoint scores. The original reheated update-48,000 checkpoint and separate decaying-learning-rate arm are not this product. Parameters/gradients/Adam are FP32 during training; EMA accumulation and reported inference are FP64. This is one selected seed, not a multi-seed uncertainty estimate or convergence certificate.

## Evaluation scope

The evaluation role comprises 282 waveforms in 266 physical-system groups: 109 SXS waveforms (106 groups) and 173 RIT waveforms (163 groups), with cross-catalogue groups counted once in the total. Fresh scoring of this constant-learning-rate update-48,000 EMA gives SXS physical-group mean strain error **0.02791640** and strain-derived curvature error **0.14532515**; RIT group-mean strain error is **0.06501308**. All 282 rows are retained with no numerical failures. At 120 solar masses, the full SXS panel has physical-group mean detector mismatch **0.1190500%**, waveform-level median **0.02167471%**, and waveform-level P95 **0.6231943%**. These are full-panel values, not a common-support four-model comparison. Earlier update-40,000 evaluation numbers are historical and are not assigned to these weights. Validation gains do not establish uniform improvement; high-q aligned validation errors regress and the highest-q precessing checks are sparse.

Strain error profiles a common time shift and rigid azimuthal rotation over all 21 modes on `[-1000,40]M`. Qualified SXS curvature error uses derivatives of native strain on `[-160,40]M` under the strain-fitted alignment; it is not an independent SXS Psi4 extraction. Detector mismatch uses a finite time window and does not cover a complete low-frequency inspiral.

Identified upstream waveform-fitting/parent-sample families and duplicate simulations are assigned to training or validation under the adopted grouping policy. Nine current evaluation groups nevertheless entered the inherited reference-time adapter fits. Direct split disjointness does not prove all upstream information exposure has been removed; unpublished upstream identities remain uncertain.

## Limits of use

The native `[-1100,80]M` support includes guard intervals outside the usual reported `[-1000,40]M` strain window (curvature is assessed on `[-160,40]M`). Native support is not an accuracy guarantee at its endpoints: the full-window example can show endpoint oscillations and large analytic-curvature excursions. The plotting command retains these features without filtering or silently cropping them.


- **Finite duration:** the model cannot generate times outside `[-1100, 80]M`. Lower-mass full-band applications require an earlier inspiral and an independently checked joining procedure.
- **Reference-frame accuracy:** headline aligned errors do not guarantee unaligned waveform accuracy. Residual time and azimuth offsets remain; this model has not established readiness for parameter estimation in a fixed physical input frame.
- **Sparse high-mass-ratio NR checks:** q4–6 precessing evaluation has measurable coverage, but q6–8 has only one physical group and q>8 has none. Synthetic pretraining extends farther, with nonuniform spin and mode coverage. The API's mathematical input domain is wider than the scientifically tested domain.
- **Curvature:** the final product has no explicit Psi4 loss term. Independent native BAM curvature checks are uneven, and are not an accuracy certificate for this checkpoint. Exact analytic differentiation does not provide a derivative-accuracy guarantee.
- **Physical coverage:** the inputs encode mass ratio and two spins, with no eccentricity or matter parameters. No uniform accuracy guarantee over the physical spin ball, extreme mass ratios, or untested physical systems follows from accepting those inputs.

## License and scientific credit

Package 0.3.1 additionally provides optional private-NR fine-tuning, documented
in [docs/FINETUNING.md](docs/FINETUNING.md). It starts from these published EMA
weights with a new optimizer and uses fixed-reference strain targets prepared
by the caller, plus frozen original-model replay. Full-network updates are the
default; finite retention panels monitor drift without a default hard bound.
They do not certify global retention or new NR accuracy. It is not the historical multi-stage training pipeline. Adapted
weights carry separate metadata and retain the architecture and symmetry
projection; the published accuracy numbers above apply only to the unchanged
v0.2.1 weights. Fine-tuning can degrade other regions or analytic curvature and
requires independent validation of the intended application.

The code, documentation, and accompanying inference weights are licensed under MIT. Preserve the copyright and permission notice in [LICENSE](LICENSE): Copyright (c) 2026 Jameson Dong and contributors. Installed dependencies retain their own licenses.

Scientific use should cite the manuscript and the relevant upstream sources it identifies: the SXS and RIT numerical-relativity catalogues; NRSur7dq4 and the surrogate modeling methods; SEOBNRv5PHM and its antisymmetric-mode extension; and BHPTNRSur2dq1e3. Cardiff–BAM supplies the auxiliary native-curvature examples. LALSuite/IMRPhenomTPHM supplies a comparison model, not training targets for this checkpoint. This inference bundle does not redistribute those catalogues, teacher evaluators, or their model files.

## Experimental Apple backend

MPS uses FP32 real/imaginary arithmetic and CPU FP64 frozen replay targets for
private fine-tuning. Published FP64 accuracy/speed results are not Metal
benchmarks. Actual Mac hardware validation is pending; see docs/METAL.md.
