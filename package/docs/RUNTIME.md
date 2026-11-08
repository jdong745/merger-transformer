# Inference execution and timing

The eager interface is the portable, differentiable reference. Use float64 to match the manuscript's EMA evaluation. The optional `compile_inference()` wrapper is forward-only, freezes a copy of the selected model, and keeps seven physical inputs and the time array as runtime arguments. It requires the tested PyTorch 2.12.x compilation API. It does not download weights or compile a hard-coded physical system.

The CPU path compiles the full field evaluation. The CUDA path compiles coefficient and basis components and performs their complex matrix multiplication outside those components. This is the measured implementation, rather than a claim that every operation is fused into one kernel. The API's domain checks occur outside the compiled numerical core. Unsupported compiler/runtime environments fail explicitly.

Compilation happens on first use and may repeat for a new input batch shape, dtype, thread setting, device, or grid length. It includes code generation, C++/GPU kernel build and first-call initialization. Run all intended shapes and several additional warmup calls before measuring steady-state inference. Do not include those first-call costs in steady-state latency. A standalone compiler-build time cannot be inferred by subtracting an eager run from the first compiled call.

A complete waveform call includes all four symmetry evaluations, all 21 signed modes and a newly constructed quintic basis. The paper benchmark uses 2361 times on `[-1100,80]M` at 0.5M spacing, FP64 EMA weights, no host-to-device copy inside the device-resident timing, and explicit GPU synchronization. Model loading is outside the timing. CPU thread count, physical batch size and microbatch size are distinct settings. If using `evaluate_coefficients`, cached coefficients or a different grid, label that changed timing contract separately.

The eager `.waveforms()` call shares one coefficient-network evaluation between strain and curvature. The compiled `.waveforms()` currently performs separate full strain and curvature calls. Do not compare their cost with a strain-only benchmark as if they were identical work.

Use the eager model for automatic differentiation. The compiled forward wrapper rejects gradient-requiring physical inputs or times. Analytic spline derivative evaluation is still provided by `evaluate(..., derivative=1/2)` and `.psi4()`, but that does not imply a compiled input-gradient or Hessian interface.

The release verification report records finite input/output checks against the selected research checkpoint and compilation controls. Paper timings describe specific hardware/software and are not a performance guarantee on another system. FP32 is an explicit optional eager choice; it is not interchangeable with the paper's FP64 precision contract.
