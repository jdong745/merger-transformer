# Merger Transformer

A standalone PyTorch interface to the extended-window Transformer for precessing binary-black-hole merger waveforms (`width=224`, `512` two-sided bump quintic B-spline knots, `2,564,276` parameters, `STEP064000` `float64` EMA weights). The model generates all 21 signed spherical-harmonic modes for `ell = 2, 3, 4` on `[-1620, 160]M` with exact parameter and temporal derivatives via automatic differentiation.

## Installation and weights

Install from this `package/` directory:

```bash
python -m pip install .
```

The release bundle separates code and weights:

```text
merger-transformer-v0.5.0/
  package/                 # this directory
  weights/weights.pt       # STEP064000 float64 EMA inference weights
```

## Generate a waveform

```python
import torch
from merger_transformer import WaveformModel, MODES

model = WaveformModel.from_pretrained("../weights/weights.pt", device="cpu")

# (q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)
x = torch.tensor([2.0, 0.2, -0.1, 0.3, -0.1, 0.05, 0.2], dtype=torch.float64)
t = torch.linspace(-1500.0, 100.0, 3201, dtype=torch.float64)
h = model(x, t)                       # complex128 [21, 3201]
h22 = h[MODES.index((2, 2))]

# Evaluates the coefficient network once for both strain and Psi4.
h, psi4 = model.waveforms(x, t)
by_mode = model.mode_dict(x, t, field="strain")
assert by_mode[(2, 2)].shape == (3201,)
```

Default precision is `float64` on CPU/CUDA and `float32` on MPS. Inputs may have shape `[7]` or `[batch, 7]`; outputs have shape `[21, T]` or `[batch, 21, T]`. Modes are ordered by ascending `ell = 2,3,4`, then ascending `m = -ell,...,ell`. Use `microbatch_size` to control physical-input chunking (default `4`).

## Inputs, frame, and units

Inputs are `(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)` with `q = m1/m2 > 0` and dimensionless spins `chi_a = S_a/m_a**2` (`|chi_a| <= 1`). Spins and inertial modes use the horizon triad `(n, Lhat × n, Lhat)` at fixed dimensionless reference frequency:

```text
M f_ref = 0.01182117827433904
f_ref   = 20 Hz × (120 M_sun / M)
```

Time is dimensionless `tau = t_physical / M_seconds` within **`[-1620, 160]M`** (`M_seconds = G M / c**3`); primary paper evaluations score `[-1500, 100]M`. Strain modes are distance-scaled `H_lm = (R / M_length) h_lm` (`M_length = G M / c**2`) with `h = h_plus - 1j * h_cross`; multiply by `M_length / D` for physical dimensionless strain. Analytic curvature is `P_lm = -d²H_lm/dtau²`; multiply by `(M_length / D) / M_seconds**2` for inverse seconds squared.

## Derivatives and reusable coefficients

```python
x = x.detach().requires_grad_(True)
t = t.detach().requires_grad_(True)
h = model(x, t)
loss = h.abs().square().mean()
grad_x, grad_t = torch.autograd.grad(loss, (x, t))

c = model.coefficients(x)                       # [21, 512]
h = model.evaluate_coefficients(c, t, derivative=0)
dh_dtau = model.evaluate_coefficients(c, t, derivative=1)
d2h_dtau2 = model.evaluate_coefficients(c, t, derivative=2)
psi4 = -d2h_dtau2
```

Weights remain frozen while parameter and time gradients are preserved (`evaluate_coefficients(..., derivative=2)` returns `+d²h/dtau²`; `psi4` applies the minus sign).

## Observer projection and plots

```bash
python -m pip install '.[plot]'
python -m merger_transformer.plotting ../weights/weights.pt --config examples/plot_config.json --output output/example
```

Add `"mass_msun"` and `"distance_mpc"` to the JSON for SI physical units (detector-frame total mass and luminosity distance). See [docs/PLOTTING.md](docs/PLOTTING.md).

For tensor projection without Matplotlib:

```python
from merger_transformer.projection import project_modes, physical_scales
h_observer = project_modes(h, inclination=1.0, azimuth=0.3)
h_plus, h_cross = h_observer.real, -h_observer.imag
M_seconds, strain_scale, psi4_scale = physical_scales(120.0, 1000.0)
```

## Compiled forward inference

On CPU or CUDA with PyTorch 2.12.x, `model.compile_inference()` creates an independent frozen Inductor forward wrapper:

```python
fast = model.compile_inference()
with torch.no_grad():
    h_fast = fast(x.detach(), t.detach())
    p_fast = fast.psi4(x.detach(), t.detach())
```

Compile and warm up each batch/grid shape before timing. On `3561` points (`[-1620, 160]M` at `0.5M` spacing) in compiled `float64` after warm-up, batched inference reaches **445 waveforms/s** (`2.25 ms/waveform`, batch `1024`) on an NVIDIA A100-SXM4-40GB GPU and **4.13 waveforms/s** (`242.3 ms/waveform`, batch `32`, `4` threads) on an AMD EPYC 7773X CPU. See [docs/RUNTIME.md](docs/RUNTIME.md).

## Examples, fine-tuning, and tests

- Private-NR fine-tuning: `python -m pip install '.[finetune]'` and see [docs/FINETUNING.md](docs/FINETUNING.md).
- Apple Silicon (experimental FP32 MPS): see [docs/METAL.md](docs/METAL.md).
- Quick examples: `python examples/generate.py ../weights/weights.pt` and `python examples/gradients.py ../weights/weights.pt`.
- Test suite: `python -m pip install '.[test]' && python -m pytest tests --weights ../weights/weights.pt -q`.

## Citation and license

Jameson Dong, *A high-precision Transformer model for precessing binary black-hole merger waveforms calibrated to numerical relativity*. MIT License ([LICENSE](LICENSE)).
