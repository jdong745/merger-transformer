# Merger Transformer

PyTorch inference for precessing binary-black-hole merger waveforms. Given a mass ratio and two spin vectors, the model generates all **21 signed strain modes through ell=4**, analytic time derivatives, and strain-derived Psi4. The eager interface supports automatic differentiation with respect to the seven physical inputs and time.

This release contains inference code and trained weights under the **MIT license**. It does not distribute training code, NR data, synthetic teacher caches, or optimizer states.

## Download and run

The easiest option is the [complete v0.2.1 bundle](https://github.com/jdong745/merger-transformer/releases/download/v0.2.1/merger-transformer-v7-complete.zip), which includes code, weights, and plotting examples. Extract it and follow `QUICKSTART.md`.

Alternatively, clone the repository and download the weights:

```bash
git clone https://github.com/jdong745/merger-transformer.git
cd merger-transformer
python -m pip install './package[plot]'
curl -L --fail https://github.com/jdong745/merger-transformer/releases/download/v0.2.1/weights.pt -o weights/weights.pt
```

Generate strain and analytic curvature:

```python
import torch
from merger_transformer import WaveformModel, MODES

torch.set_num_threads(1)
model = WaveformModel.from_pretrained("weights/weights.pt", device="cpu")
x = torch.tensor([2., .2, -.1, .3, -.05, .1, -.2], dtype=torch.float64)
t = torch.linspace(-1000., 40., 1041, dtype=torch.float64)
h, psi4 = model.waveforms(x, t)  # each [21, 1041], complex128
h22 = h[MODES.index((2, 2))]
```

Plot a configurable example:

```bash
python -m merger_transformer.plotting weights/weights.pt --config package/examples/plot_config.json --output output/example
```

This writes mode plots, observer polarizations, Psi4 plots, and the underlying arrays. For gradients, batching, observer projection, physical-unit conversion, and optional frozen CPU/CUDA compilation, see the [package documentation](package/README.md), [plotting guide](package/docs/PLOTTING.md), and [runtime guide](package/docs/RUNTIME.md).

## Physical conventions

- Input: `(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)` with `q=m1/m2>0`.
- Spins and modes use the fixed inertial reference triad `(n, Lhat × n, Lhat)` at `M f_ref = 0.01182117827433904`, equivalent to 20 Hz at 120 solar masses.
- Native time support is `[-1100,80]M`; reported strain comparisons use `[-1000,40]M`. Guard-interval support is not an endpoint-accuracy guarantee.
- Strain modes are distance-scaled: `H_lm=(R/M) h_lm`, with `h=h_plus-i h_cross`. Curvature is `-d²H_lm/dtau²`.
- The interface does not transport spins, fit a time/azimuth correction, or extend the signal with a QNM tail.

Read the [model card](package/MODEL_CARD.md) for training provenance, evaluated domains, and limitations. These finite-window waveforms do not by themselves cover a full detector-band inspiral. High-q precessing evaluation remains sparse.

## Selected model

Version **0.2.1** contains the width-192 v7 Transformer with **1,974,036 parameters**, selected at update **48,000** using float64 EMA weights. Updates 40,000–48,000 used constant learning rate `1e-5`. The release contains this single model.

The release passed 27 installed-package checks, source-versus-package parity, and CPU/CUDA frozen-compilation correctness controls. Compilation is optional and forward-only; use the eager path for input/time automatic differentiation.

To run the supplied checks:

```bash
python -m pip install './package[test]'
python -m pytest package/tests --weights weights/weights.pt -q
```

## Citation and license

Jameson Dong, *A Transformer model for precessing binary black-hole merger waveforms calibrated to numerical relativity*. See [CITATION.cff](CITATION.cff).

Code, documentation, and accompanying inference weights are MIT licensed. Numerical-relativity and teacher-model provenance is described in the paper and model card. Dependencies retain their own licenses.
