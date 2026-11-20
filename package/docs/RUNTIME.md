# Inference execution and runtime

The eager interface (`WaveformModel`) is the portable reference supporting `float64`/`complex128` execution and automatic differentiation with respect to inputs and time.

## Compiled forward inference (`float64`)

With PyTorch 2.12.x on CPU or CUDA, `model.compile_inference()` creates an independent frozen TorchInductor forward snapshot (`fullgraph=True`, `dynamic=False`, `config.freezing=True`). Physical parameters and time grids remain runtime arguments, and no basis matrix is cached across calls.

The first call for a new `(batch_size, time_len, device, dtype)` triggers JIT compilation and should be warmed up prior to timing. Each timed call evaluates the full parameter-to-strain pipeline, including four-way symmetry projection and 512-knot B-spline evaluation across all 21 signed modes at all `3561` time points on `[-1620, 160]M` (`0.5M` spacing). Steady-state warm-started compiled `float64` inference achieves:

- **GPU (NVIDIA A100-SXM4-40GB, batch `1024`, microbatch `128`)**: **`445 waveforms/s`** (`2.25 ms/waveform`, `2.301 s` total, `9.68 GiB` peak memory).
- **CPU (AMD EPYC 7773X, `4` threads, batch `32`)**: **`4.13 waveforms/s`** (`242.3 ms/waveform`, `7.75 s` total).

Across all tested CPU and GPU parity controls (including perturbed runtime time grids and discrete symmetry checks), compiled execution matches eager `float64` and autograd checks within `3.8e-15` relative norm with zero graph breaks.

## Precision and differentiation

- Compiled inference is forward-only and rejects gradient-requiring inputs (`requires_grad=True`). Use the eager `WaveformModel` for input/time automatic differentiation.
- Analytic spline derivatives (`derivative=1, 2` and `.psi4()`) remain available on both eager and compiled paths. Eager `.waveforms()` computes the coefficient network once for both strain and `Psi4`.
- On Apple Silicon (Metal/MPS), eager `float32` real-channel inference is provided; see [METAL.md](METAL.md).
