# Apple Silicon / Metal

An experimental PyTorch MPS path supports inference and full-network private-NR
fine-tuning using the **same published weights**. It uses FP32 real/imaginary
channels: Metal does not provide the FP64 reference arithmetic. CPU/CUDA keep
their existing FP64 default. No weights are retrained or replaced by installing
this version.

**Verification status:** the real-channel waveform, symmetry, derivative and
training arithmetic is tested on Linux CPU. Actual Apple GPU execution and
performance have not yet been verified by the maintainers. Run the hardware
check below before using this experimental backend for research. Passing a
short check does not establish equivalence of long fine-tuning trajectories.

On six fixed synthetic CPU inputs spanning q=1 to 1000, FP32 real-channel
outputs differed from FP64 by at most 3.06e-6, 4.72e-6 and 5.19e-6 in
per-input all-mode relative norm for h, h' and h'', respectively. This includes
near-symmetric inputs and spline knots, but is not a Metal measurement or a
bound over the parameter domain. The unchanged CPU FP64 path reproduces the
previous package output bit for bit on a separate four-input control.

## Install

Use native arm64 Python on Apple Silicon, a macOS version supported by your
installed PyTorch, and a current MPS-capable PyTorch build. See the
[official MPS instructions](https://docs.pytorch.org/docs/stable/notes/mps.html).
From the repository root:

```bash
python -m pip install './package[finetune,plot]'
python -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'
```

Download the separately released `weights.pt` as described in the main README.
`is_available()` must print `True`. The package does not enable implicit CPU
operator fallback. Explicit FP64 on `device="mps"` is rejected; use CPU for FP64.
The frozen Inductor compilation interface remains CPU/CUDA-only.

## Generate waveforms

```python
import torch
from merger_transformer import WaveformModel

model = WaveformModel.from_pretrained(
    'weights/weights.pt', device='mps', microbatch_size=1,
)
x = torch.tensor([2., .2, -.1, .3, -.05, .1, -.2], device='mps')
t = torch.linspace(-1100., 80., 2361, device='mps')
with torch.no_grad():
    h_ri, psi4_ri = model.waveforms_ri(x, t)
    # MPS float32, shape [21,2361,2]; last axis is (real, imaginary).
```

`coefficients_ri`, `strain_ri`, `psi4_ri`, `waveforms_ri` and
`evaluate_coefficients_ri` retain tensors on the model's device. The latter
accepts `derivative=0,1,2`; derivative 2 is **+h''**, whereas `psi4_ri` returns
**-h''**. All four symmetry branches and all 21 signed modes are retained.
The existing complex interfaces (`model(x,t)`, `waveforms`, `psi4`,
`coefficients`) explicitly copy real channels to CPU and return complex64.
This transfer retains autograd, but incurs synchronization/copy overhead.

```python
x.requires_grad_()
h = model.strain_ri(x, t)
gradient = torch.autograd.grad(h.square().sum(), x)[0]
```

Analytic spline derivatives use the same basis, rounded to FP32. Check errors
against CPU FP64 for your application, particularly weak modes and parameter
derivatives. No Metal Hessian or long-run optimizer reproducibility claim is
made. The plotting example accepts `--device mps`; complex projection and
matplotlib then run on CPU.

## Fine-tune

The [fixed-reference private-data contract](FINETUNING.md) is unchanged: supply
compatible strain, input spins at the specified reference event, and your own
group-separated validation set. There is no automatic spin transport.

```bash
merger-transformer-finetune \
  --weights weights/weights.pt \
  --manifest private/manifest.json \
  --output private/adapted_mps \
  --device mps --steps 2000
```

`--dtype auto` selects FP32 for MPS. All network parameters are trainable by
default. The loss, 1:9 private/replay sampling, equal weights for the two averaged
losses, validation selection and optional retention limit are unchanged.
The student, backward pass, AdamW and EMA use FP32 on MPS. The original frozen
retention teacher runs on **CPU FP64**, generating a local cache once. This
preparation can take time; it is not Metal training throughput. Targets are
converted to FP32 real channels when the student evaluates a row.

Start with microbatches of one (the trainer already does this). Attention and
optimizer state use shared system memory. Do not override OS/PyTorch memory
limits to force a larger run. Resume on the same backend and settings with
`--resume --steps NEW_TOTAL`; cross-device continuation is not supported.
Exported adapted weights can be loaded on CPU/CUDA/MPS. Exporting FP32-trained
parameters as FP64 does not recover lost precision.

## Verify on your Mac

From the repository root, with `PYTORCH_ENABLE_MPS_FALLBACK` unset:

```bash
python package/examples/verify_metal.py \
  --weights weights/weights.pt --output private/metal_check --finetune
```

This checks all-mode strain and first/second derivatives against CPU FP64,
parameter/time gradients, near-symmetric and high-q inputs, batch consistency,
exchange/reflection, CPU complex output, and a two-update synthetic full-network
fine-tune with replay, save/resume and reload. `REPORT.json` records results and
environment. Its numerical thresholds are smoke-test tolerances, not model
accuracy claims. On a machine without MPS the default command fails explicitly.
`--device cpu`/`cuda` runs a separate arithmetic control and records
`hardware_mps_verified: false`; it does not verify Metal.
