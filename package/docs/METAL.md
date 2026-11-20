# Apple Silicon / Metal

An experimental PyTorch MPS path supports `float32` real/imaginary-channel inference and full-network private-NR fine-tuning using the released `v0.5.0` weights. CPU and CUDA keep the `float64` reference path.

**Verification status:** real-channel waveform, symmetry, derivative, and fine-tuning arithmetic has been verified on Linux CPU against `float64` reference outputs. Actual Apple GPU execution remains unverified on Mac hardware; run `verify_metal.py` on Apple Silicon before deploying MPS runs.

On six synthetic CPU inputs spanning `q = 1` to `1000` over `[-1620, 160]M`, `float32` real-channel evaluations matched `float64` reference outputs within `2.17e-6` (`h`), `3.54e-6` (`dh/dtau`), and `8.30e-6` (`d^2h/dtau^2`) in relative norm, with `0.0` batch discrepancy, `0.0` discrete `E/P/EP` symmetry error, and parameter/time gradients agreeing within `7.57e-6`.

## Install

Use native `arm64` Python on Apple Silicon with an MPS-capable PyTorch build. From the repository root:

```bash
python -m pip install './package[finetune,plot]'
python -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'
```

Download `weights.pt` from release `v0.5.0` into `weights/weights.pt`. Implicit CPU operator fallback is not enabled; `device="mps"` requires `float32` (use `device="cpu"` for `float64`).

## Generate waveforms

```python
import torch
from merger_transformer import WaveformModel

model = WaveformModel.from_pretrained("weights/weights.pt", device="mps", microbatch_size=1)
x = torch.tensor([2., .2, -.1, .3, -.05, .1, -.2], device="mps")
t = torch.linspace(-1500., 100., 3201, device="mps")
with torch.no_grad():
    h_ri, psi4_ri = model.waveforms_ri(x, t)
    # MPS float32, shape [21, 3201, 2]; last axis is (real, imaginary).
```

`coefficients_ri`, `strain_ri`, `psi4_ri`, `waveforms_ri`, and `evaluate_coefficients_ri` keep tensors on the model device (`derivative=2` returns `+h''`, while `psi4_ri` returns `-h''`). Complex convenience methods (`model(x, t)`, `waveforms`, `psi4`, `coefficients`) explicitly return `complex64` tensors on CPU while preserving autograd.

```python
x.requires_grad_()
h = model.strain_ri(x, t)
gradient = torch.autograd.grad(h.square().sum(), x)[0]
```

## Fine-tune

The [private-NR contract](FINETUNING.md) is unchanged (`--window balanced` scores `[-1500, -160)M` and `[-160, 100]M`):

```bash
merger-transformer-finetune \
  --weights weights/weights.pt \
  --manifest private_data/manifest.json \
  --output finetune_runs/adapted_mps \
  --device mps --steps 2000
```

Student parameters, AdamW, and EMA run in `float32` on MPS while the unchanged frozen replay teacher prepares targets on CPU `float64`. Resume on the same backend with `--resume --steps NEW_TOTAL`. Exported weights load identically on CPU, CUDA, and MPS.

## Verify on Apple Silicon

From the repository root, with `PYTORCH_ENABLE_MPS_FALLBACK` unset:

```bash
python package/examples/verify_metal.py \
  --weights weights/weights.pt --output finetune_runs/metal_check --finetune
```

This checks all-mode strain, first/second derivatives, parameter/time gradients, near-symmetric and high-`q` inputs, batch consistency, `E/P/EP` equivariance, CPU complex outputs, and a two-step synthetic fine-tune with replay, save/resume, and reload.
