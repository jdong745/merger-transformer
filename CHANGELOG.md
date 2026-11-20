# Changes

## 0.5.0

- Release the extended-window width-224 Merger Transformer (`2,564,276` parameters, `512` two-sided bump quintic B-spline coefficients, `518` stored binary64 knots, `STEP064000` `float64` EMA weights).
- Extend native time support from `[-1100, 80]M` to **`[-1620, 160]M`** (`3561` samples at `0.5M` spacing) and primary evaluation, plotting defaults, examples, and private-NR fine-tuning windows to **`[-1500, 100]M`**.
- Update `ModelConfig`, `_basis.py`, `api.py`, `finetune.py` (`STATE_FORMAT = merger-transformer-finetune-state-v3`), and the test suite cleanly for the 512-knot bump architecture while preserving all inference, autograd, compiled Inductor, Metal/MPS, and replay fine-tuning capabilities.
- Record compiled batched `float64` warm-started runtime benchmarks over `3561` time samples (`445 waveforms/s` on A100 GPU at batch `1024`; `4.13 waveforms/s` on 4 AMD EPYC 7773X CPU threads at batch `32`).

## 0.4.0

- Add experimental Apple Metal/MPS FP32 inference and full-network fine-tuning via real/imaginary channels, with explicit CPU complex output and CPU FP64 replay targets.
- Add device-resident real-channel APIs, numerical controls and a Mac hardware verification script.

## 0.3.1

- Default to full-network adaptation at learning rate `1e-6`, with nine frozen original-model samples per private NR row and equal weights for the two separately averaged losses.
- Add deterministic replay caches and independent q-stratified retention validation.

## 0.3.0

- Add optional local private-NR strain fine-tuning with head-only or full-network updates, masked signed-mode targets, group-separated validation, EMA exports, and optimizer/sampler resume.
