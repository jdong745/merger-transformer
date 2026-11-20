# Merger Transformer

PyTorch inference and local fine-tuning for precessing binary-black-hole merger waveforms. Given a mass ratio and two spin vectors, the model predicts 512 quintic B-spline coefficients for all **21 signed spherical-harmonic strain modes through ell=4** over `[-1620, 160]M`, yielding `C^4` waveforms with exact parameter and temporal derivatives via automatic differentiation (`Psi4 = -d²H/dtau²`).

Code and inference weights are **MIT licensed**. NR waveforms, synthetic teacher caches, and training optimizer states are not distributed.

## Download and run

Download the [complete v0.5.0 bundle](https://github.com/jdong745/merger-transformer/releases/download/v0.5.0/merger-transformer-v0.5.0-complete.zip) (code, weights, examples, and quickstart), extract it, and follow `QUICKSTART.md`.

Alternatively, clone the repository and download the weights:

```bash
git clone https://github.com/jdong745/merger-transformer.git
cd merger-transformer
python -m pip install './package[plot]'
curl -L --fail https://github.com/jdong745/merger-transformer/releases/download/v0.5.0/weights.pt -o weights/weights.pt
```

Generate strain and analytic curvature across `[-1500, 100]M`:

```python
import torch
from merger_transformer import WaveformModel, MODES

torch.set_num_threads(1)
model = WaveformModel.from_pretrained("weights/weights.pt", device="cpu")
x = torch.tensor([2., .2, -.1, .3, -.05, .1, -.2], dtype=torch.float64)
t = torch.linspace(-1500., 100., 3201, dtype=torch.float64)
h, psi4 = model.waveforms(x, t)  # each [21, 3201], complex128
h22 = h[MODES.index((2, 2))]
```

Plot a configurable example:

```bash
python -m merger_transformer.plotting weights/weights.pt --config package/examples/plot_config.json --output output/example
```

This writes mode plots, observer polarizations, Psi4 plots, and the underlying arrays. For gradients, batching, observer projection, physical-unit conversion, and frozen CPU/CUDA compilation, see the [package documentation](package/README.md), [plotting guide](package/docs/PLOTTING.md), and [runtime guide](package/docs/RUNTIME.md).

## Model summary (v0.5.0)

- **Architecture & weights**: width-224 extended-window Transformer (`2,564,276` parameters) with `512` two-sided bump quintic B-spline coefficients (`518` stored `float64` knots) selected at update **64,000** (`float64` EMA).
- **Inputs**: `(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)` with `q = m1/m2 > 0`, enforcing exact binary-exchange and equatorial-reflection equivariance via four-way group projection.
- **Reference frame**: spins and modes use the inertial horizon triad `(n, Lhat × n, Lhat)` at `M f_ref = 0.01182117827433904` (20 Hz at `120 M_sun`).
- **Time window**: native support `[-1620, 160]M`, primary evaluation window `[-1500, 100]M`.
- **Evaluation highlights (`[-1500, 100]M`)**: mean balanced relative strain error `0.0305` and `120 M_sun` detector mismatch `0.1536%` across `109` SXS evaluation waveforms (`106` physical systems); `0.0184` strain error (`0.0253%` mismatch) inside the `NRSur7dq4v2` calibration domain (`69/68`).
- **Compiled `float64` throughput (`3561` samples, after warm-up)**: `445 waveforms/s` (`2.25 ms/waveform`, batch `1024`) on an NVIDIA A100-SXM4-40GB GPU, and `4.13 waveforms/s` (`242.3 ms/waveform`, batch `32` on `4` threads) on an AMD EPYC 7773X CPU.

Read the [model card](package/MODEL_CARD.md) for training provenance, evaluated regimes, and limitations.

## Fine-tune with private NR

Install the optional trainer and prepare a local manifest with separate physical-system groups for training and validation:

```bash
python -m pip install './package[finetune]'
merger-transformer-finetune --weights weights/weights.pt \
  --manifest private_data/manifest.json --check-data
merger-transformer-finetune --weights weights/weights.pt \
  --manifest private_data/manifest.json --output finetune_runs/my_nr \
  --device cuda --steps 2000
```

The default updates the **full shared network** at learning rate `1e-6`, with nine frozen original-model replay samples per private NR row on the `[-1500, 100]M` balanced/merger windows. Private and replay mean losses are equally weighted. `BEST` is selected by private-NR validation error, with an optional `--max-replay-l2` stratum bound. See the [fine-tuning guide](package/docs/FINETUNING.md) for the prepared data contract, conventions, and resume commands.

## Tests

```bash
python -m pip install './package[test]'
python -m pytest package/tests --weights weights/weights.pt -q
```

### Apple Silicon

Experimental Metal/MPS inference and full-network private fine-tuning are available through the [Metal guide](package/docs/METAL.md) (`float32` real/imaginary channels; CPU/CUDA keep the `float64` reference path).

## Citation and license

Jameson Dong, *A high-precision Transformer model for precessing binary black-hole merger waveforms calibrated to numerical relativity*. See [CITATION.cff](CITATION.cff).

Code, documentation, and accompanying inference weights are MIT licensed. Numerical-relativity and teacher provenance are documented in the paper and model card.
