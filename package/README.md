# Merger Transformer

A standalone PyTorch interface to the v7 Transformer for precessing binary-black-hole merger waveforms. The model generates all 21 signed spherical-harmonic modes with degree `ell = 2, 3, 4`, together with differentiable spline coefficients and analytic time derivatives.

This package contains inference code and an optional local private-NR fine-tuning tool. The accompanying published weight file contains the selected constant-low-learning-rate step-48,000 exponential-moving-average (EMA) parameters and fixed spline knots. The optional trainer, optimizer state, NR waveforms, synthetic teacher arrays, and waveform-generator dependencies are not required for inference.

## Installation and weights

Install from this code directory:

```bash
python -m pip install .
```

The code and weights are distributed separately within the bundle:

```text
bundle/
  package/                 # this directory
  weights/weights.pt       # selected v7 inference weights
```

The examples below run from `package/`. Pass the appropriate path if the weights are stored elsewhere. Loading uses the supplied file and does not download models or data.

## Generate a waveform

```python
import torch
from merger_transformer import WaveformModel, MODES

model = WaveformModel.from_pretrained("../weights/weights.pt", device="cpu")

# q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z
x = torch.tensor([2.0, 0.2, -0.1, 0.3, -0.1, 0.05, 0.2],
                 dtype=torch.float64)
t = torch.linspace(-1000.0, 40.0, 1041, dtype=torch.float64)
h = model(x, t)                       # complex128 [21, 1041]
h22 = h[MODES.index((2, 2))]

# Computes the coefficient network once for both fields.
h, psi4 = model.waveforms(x, t)
by_mode = model.mode_dict(x, t, field="strain")
assert by_mode[(2, 2)].shape == (1041,)
```

The default inference precision is float64 on CPU/CUDA and float32 on the experimental MPS backend. Inputs may have shape `[7]` or `[batch, 7]`; the corresponding outputs have shape `[21, time]` or `[batch, 21, time]`. The one-dimensional time grid is shared across the batch. Mode order is ascending `ell`, then ascending `m`: `(2,-2), ..., (2,2), (3,-3), ..., (3,3), (4,-4), ..., (4,4)`.

On a PyTorch installation with CUDA support, use `device="cuda"` and create inputs and times on that device. `from_pretrained(..., microbatch_size=4)` sets the number of physical inputs processed together; four is the default. All four symmetry evaluations remain part of each physical waveform.

## Inputs, frame, and units

The seven inputs are `(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)`, where `q = m1/m2 > 0` and each spin is dimensionless, `chi_a = S_a/m_a**2`. Inputs must be finite and each spin magnitude must be at most one. There is no mass-ratio sorting or artificial mass-ratio cutoff. For `q < 1`, body 1 is the lighter body and its spin still occupies the first spin slot.

Specify the spins at fixed dimensionless reference frequency

```text
M f_ref = 0.01182117827433904
f_ref = 20 Hz × (120 solar masses / M)
```

Here the NR reference frequency is the horizon orbital frequency divided by pi. The spin components and output modes use the inertial triad `(n, Lhat × n, Lhat)` fixed at that reference epoch, with `n` pointing from body 2 to body 1. The interface does not evolve spins from another frequency or construct this reference frame for the caller.

The time argument is dimensionless `tau = t_physical / M_seconds`, with support **`[-1100, 80]`**, including both endpoints. `M_seconds = G M / c**3`. Arbitrary sample times inside this interval are supported; extrapolation beyond it is rejected. The learned clock follows the NR target curvature-energy time moment. It is not a guarantee that the strain peak or reference-frequency event occurs at `tau = 0`. Inference applies no automatic peak centering or time/azimuth alignment.

The returned strain modes are the distance-scaled dimensionless quantities

```text
H_lm = (R / M_length) h_lm_physical,    M_length = G M / c**2.
```

Thus, for luminosity distance `D` expressed in the same length units as `M_length`, multiply the returned modes by `M_length / D` to obtain physical dimensionless strain. The complex polarization convention is `h = h_plus - 1j * h_cross`; sum the modes against spin-weight `-2` spherical harmonics for an observer direction. Detector response is a subsequent operation. The network's amplitude factor is already included and requires no additional normalization.

`model.psi4(x, t)` returns `P_lm = -d²H_lm/dtau²`. To express the same sign convention as `-d²h_physical/dt_physical²` in inverse seconds squared, multiply `P_lm` by `(M_length / D) / M_seconds**2`. It is curvature derived from the generated strain, not an independent curvature prediction.

## Derivatives and reusable coefficients

```python
x = x.detach().requires_grad_(True)
t = t.detach().requires_grad_(True)
h = model(x, t)
loss = h.abs().square().mean()
grad_x, grad_t = torch.autograd.grad(loss, (x, t))

c = model.coefficients(x)                       # [21, 384]
h = model.evaluate_coefficients(c, t, derivative=0)
dh_dtau = model.evaluate_coefficients(c, t, derivative=1)
d2h_dtau2 = model.evaluate_coefficients(c, t, derivative=2)
psi4 = -d2h_dtau2
```

Model weights are frozen while input-parameter and time gradients remain available. `evaluate_coefficients(..., derivative=2)` returns the **second derivative of strain without a sign reversal**; the separate `psi4` methods apply the minus sign. Batched coefficient arrays have shape `[batch, 21, 384]`. `mode_dict(..., field="psi4")` provides the curvature output indexed by `(ell, m)`.

## Observer projection and plots

Install the optional plotting dependencies, then run from this directory:

```bash
python -m pip install '.[plot]'
python -m merger_transformer.plotting ../weights/weights.pt --config examples/plot_config.json --output output/example
```

The JSON specifies the runtime seven-parameter input, inclination and azimuth in radians, and time grid. Edit it to generate another system. The command writes:

- `example_polarizations.png/.pdf`: projected strain and analytic curvature;
- `example_modes.png/.pdf`: real/imaginary parts of all 21 signed strain modes;
- `example_psi4_modes.png/.pdf`: all 21 analytic curvature modes;
- `example.npz` and `example_settings.json`: full arrays, parameters and conventions.

No time/azimuth fitting, filtering, mode truncation or peak recentering is performed. The example is a synthetic input, not an NR accuracy comparison. Add **both** `"mass_msun": 120.0` and `"distance_mpc": 1000.0` to obtain physical-unit plots. This uses detector-frame mass and luminosity distance; the NPZ retains dimensionless arrays too. It changes output units, not the spin reference epoch: the physical reference frequency is `20 Hz * 120 / mass_msun`. See [docs/PLOTTING.md](docs/PLOTTING.md).

For tensor projection without Matplotlib:

```python
from merger_transformer.projection import project_modes, physical_scales
h_observer = project_modes(h, inclination=1.0, azimuth=0.3)
h_plus, h_cross = h_observer.real, -h_observer.imag
M_seconds, strain_scale, psi4_scale = physical_scales(120.0, 1000.0)
```

Observer angles use the model's reference axes, not a fitted source azimuth. Detector antenna responses are not included.

## Optional frozen compilation

The default eager path preserves automatic differentiation. With the tested **PyTorch 2.12.x**, a CPU or CUDA model can also create an independent frozen snapshot:

```python
fast = model.compile_inference()
with torch.no_grad():
    h_fast = fast(x.detach(), t.detach())  # first call includes compilation
    p_fast = fast.psi4(x.detach(), t.detach())
```

Physical inputs remain runtime variables. Compilation is lazy and shape-specific. Compile and warm up every batch/grid shape before measuring steady-state execution. CPU compilation includes generated C++ kernel construction and requires a compatible host compiler. CUDA compilation needs a supported GPU toolchain. No prebuilt machine-specific kernels are distributed. Cold costs are separate from waveform-generation latency.

The compiled wrapper is **forward only** and rejects inputs requiring gradients. Use the eager `WaveformModel` for parameter/time gradients or differentiable inference; Hessian controls apply to CPU/CUDA, not the unverified MPS backend. Analytic `psi4` is available in both paths. CPU uses a compiled full-field graph; CUDA compiles coefficient and basis components followed by complex matrix multiplication. Neither caches the grid basis across calls. Compiled `.waveforms()` currently invokes two field evaluations; eager `.waveforms()` computes the coefficient network once. Compiler failures raise an error rather than silently substituting eager inference. Set `microbatch_size` explicitly for throughput tests; the default is four physical inputs. See [docs/RUNTIME.md](docs/RUNTIME.md).

## Scientific scope

The native `[-1100,80]M` support includes guard intervals outside the usual reported `[-1000,40]M` strain window (curvature is assessed on `[-160,40]M`). Native support is not an accuracy guarantee at its endpoints: the full-window example can show endpoint oscillations and large analytic-curvature excursions. The plotting command retains these features without filtering or silently cropping them.


The model covers a finite late-inspiral, merger, and ringdown interval. Its main reported strain comparisons use `[-1000, 40]` in total-mass units and optimize a common time shift and azimuthal rotation across modes. Those scores do not establish equivalent accuracy directly in the unaligned input frame or for a complete detector-band inspiral.

Training includes high-mass-ratio synthetic teachers, but precessing SXS evaluation above `q = 4` is sparse and has no `q > 8` systems. Accepting an input does not establish calibration there. In particular, high-mass-ratio output should be treated as extrapolation supported unevenly by the available training sources. The final product has no explicit Psi4 loss term; analytic differentiation ensures consistency with its strain representation, not curvature accuracy. See [MODEL_CARD.md](MODEL_CARD.md) for the architecture, training provenance, evaluation scope, and limitations.

## Examples and checks

For local adaptation with your own NR data, install `.[finetune]` and follow
[docs/FINETUNING.md](docs/FINETUNING.md). That guide defines the prepared data
contract, head-only or full-network updates, independent validation groups,
EMA exports, and optimizer-state resume. Adapted exports use the same inference
API but carry separate provenance; published-model accuracy claims do not apply
automatically to fine-tuned weights.

From `package/`, after installation:

```bash
python examples/generate.py ../weights/weights.pt --output waveform.pt
python examples/gradients.py ../weights/weights.pt
```

The optional checks use synthetic parameter inputs and targets; they require no NR files:

```bash
python -m pip install '.[test]'
python -m pytest tests --weights ../weights/weights.pt -q
```

They check the spline against SciPy, time derivatives against analytic curvature, parameter gradients and second derivatives, exchange/reflection symmetry, batch consistency, and physical input guards.

## Reference and license

Jameson Dong, *A Transformer model for precessing binary black-hole merger waveforms calibrated to numerical relativity*, v7 manuscript.

This bundle selects the width-192 recovery checkpoint after constant-learning-rate continuation from update 40,000 to **48,000**, using EMA weights and learning rate `1e-5` throughout those final 8,000 updates. This is the user-selected product; the earlier reheated 48,000 checkpoint and cosine-decay arm are distinct models. No additional time or azimuth correction is applied.

Code, documentation, and the accompanying inference weights are provided under the MIT License. Copyright (c) 2026 Jameson Dong and contributors; see [LICENSE](LICENSE). PyTorch and other installed dependencies retain their own licenses. The model's scientific provenance includes SXS and RIT numerical relativity and NRSur7dq4, SEOBNRv5PHM, and BHPTNRSur2dq1e3 teachers, as described in the manuscript and model card.

### Apple Silicon

Experimental Metal/MPS inference and full-network private fine-tuning are available
through the [Metal guide](docs/METAL.md). MPS uses FP32 real channels; the original
CPU/CUDA reference path remains FP64. Actual Apple hardware verification is pending.
