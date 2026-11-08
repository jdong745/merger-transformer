# Changes

## 0.3.1

- Default to full-network adaptation at learning rate 1e-6, with nine frozen original-model samples per private NR row and equal weights for the two separately averaged losses.
- Add deterministic replay caches and independent q-stratified retention validation. Select BEST by private-NR validation without a default hard drift bound; offer an optional user-set bound. Keep the original published weights unchanged.
- Record separate private/replay losses, retention tails and strata, cache/teacher resume bindings, and adapted-model provenance. NR-only 0.3.0 optimizer states cannot be resumed with this new recipe.

## 0.3.0

- Add optional local private-NR strain fine-tuning with head-only or full-network updates, masked signed-mode targets, group-separated validation, EMA exports, and optimizer/sampler resume.
- Accept adapted inference exports with explicit parent-product provenance. Retain the inference API, symmetry projection, and fixed spline basis.
- Add a synthetic end-to-end example and data, gradient, export, symmetry and resume checks.

The published v0.2.1 model weights and release archives are unchanged. No NR data or historical training optimizer states are included.
