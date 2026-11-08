# Fine-tuning with private NR waveforms

The optional trainer adapts the **released v7 constant-48,000 EMA model** to
locally supplied strain targets. It does not require the original training
catalogue, teachers, research repository, or optimizer state. The original
weights and architecture are unchanged. Adapted weights are separate models;
the paper's accuracy numbers do not apply to them automatically.

Install the current repository version, rather than the historical v0.2.1 ZIP:

```bash
python -m pip install './package[finetune]'
```

## 1. Prepare local targets

The input is **prepared dimensionless strain modes**, not raw simulation
directories or Psi4. Your extraction, junk removal, memory treatment, frame
rotation and reference-spin measurement remain your responsibility. No spins,
time origins, or phase branches are inferred from a better match to this model.

Each simulation is one NPZ file with exactly these arrays:

| Array | Shape / type | Meaning |
| --- | --- | --- |
| `params` | `[7]`, real | `(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)` |
| `times` | `[T]`, real | Strictly increasing dimensionless `tau`, within `[-1100,80]` |
| `strain` | `[21,T]`, complex | Distance-scaled `H_lm`, ordered as `merger_transformer.MODES` |
| `modes` | `[21,2]`, integer | The explicit ordered `(ell,m)` list |
| `mask` | `[T]` or `[21,T]`, boolean | True only for measured, usable samples/modes |

Different rows may have different grids. Nonuniform grids use trapezoidal
quadrature, without bridging masked gaps. A missing mode is **unknown**, not
zero: set its mask false. Do not reconstruct negative-m modes by reflection
for a generic precessing system, or discard the odd-m branch. Measured zeros
retain a true mask. Nonfinite values are permitted only behind a false mask.

The physical conventions must match the release:

- `q=m1/m2`; keep the same body ordering in the spins, orbital separation
  direction and modes. No automatic sorting is performed.
- Spins are in `(n, Lhat cross n, Lhat)` at
  `Mf_ref=0.01182117827433904`, with `n` pointing from body 2 to body 1.
  Modes must be rotated into that **same fixed inertial triad**. They are not
  co-precessing modes. A per-mode phase adjustment is not a rigid rotation.
- `H=(R/M)*(h_plus-i*h_cross)`. Use geometrized total mass M for time and
  amplitude normalization; do not supply detector strain in SI units.
- The NR time coordinate is `tau=(u-t0_curvature)/M`, **not simply time from
  the strain peak**. The published preparation uses a curvature-power moment:
  `P=sum_lm |d2H_lm/du2|^2`, `w=[P*(1-exp(-(u-u_start)^2/(2*(100M)^2)))]^8`,
  and `t0_curvature=integral(u*w du)/integral(w du)`. The original reference
  calculation uses a 0.5M grid and sixth-order finite differences. Use the
  full reliable extraction record and verify junk suppression and endpoint
  stability before cropping. A cropped window alone cannot recover this clock.
- Targets must use compatible oscillatory/memory-subtracted strain conventions.
  The release's SXS targets subtract the recorded BMS energy-flux contribution;
  RIT targets use the documented low-frequency treatment. Subtracting a constant
  from `(2,0)` is not an equivalent memory prescription. The trainer does not
  silently filter or subtract memory from your data. See the paper and
  [model card](../MODEL_CARD.md).

For already prepared arrays:

```python
import numpy as np
from merger_transformer import MODES

np.savez("private_data/run001.npz", params=params, times=tau,
         strain=H, modes=np.asarray(MODES, dtype=np.int64),
         mask=observed.astype(bool))
```

Create `private_data/manifest.json`:

```json
{
  "format": "merger-transformer-private-nr-v1",
  "reference_contract": "fixed_physical_reference_n_Lcrossn_L_v1",
  "Mf_ref": 0.01182117827433904,
  "time_coordinate": "tau=(u-t0_curvature)/M",
  "strain_convention": "H=(R/M)*(h_plus-i*h_cross)",
  "target_field": "strain",
  "memory_convention": "release_compatible_memory_subtracted",
  "rows": [
    {"id": "run001", "group": "binary001", "split": "train", "file": "run001.npz"},
    {"id": "run002", "group": "binary002", "split": "validation", "file": "run002.npz"}
  ]
}
```

Paths are relative to the manifest. Put all resolutions, reruns, and duplicate
versions of one physical system in the same group. A group cannot cross the
two splits. The validator checks declared groups, not hidden physical aliases;
identifying those is the data owner's responsibility. Keep a separate final
test set outside this manifest. At least one independent validation group is
required, but one group is not adequate evidence of generalization.

Check shapes, mode ordering, masks, physical ranges, conventions and split
overlap without training:

```bash
python -m merger_transformer.finetune --weights weights/weights.pt \
  --manifest private_data/manifest.json --check-data
```

This checks the declaration and arrays; it cannot certify that the declared
frame, memory subtraction or reference event is physically correct.

## 2. Run a bounded adaptation

```bash
python -m merger_transformer.finetune \
  --weights weights/weights.pt --manifest private_data/manifest.json \
  --output finetune_runs/private_nr --device cuda:0 \
  --steps 2000 --batch-size 4 --lr 1e-5 --eval-every 100
```

The equivalent installed command is `merger-transformer-finetune`.
Use `--device cpu --threads 1` for CPU operation. Default arithmetic is
float64/complex128; `--dtype float32` is an explicit lower-precision option.
No compilation, mixed precision, distributed training or automatic GPU
selection is used. The basis is fixed; its canonical float64 knots are restored
in exports even after float32 training. Exporting parameters as float64 does
not recover precision lost during float32 training.

Default `--trainable head` updates the shared linear output and mode-coupling
head, freezing the conditioner and coefficient-attention backbone.
`--trainable all` updates the full shared network. Both retain the four
exchange/reflection branches, all 21 signed output modes, and the same quintic
time regularity. Fine-tuning does not itself guarantee accurate curvature.

Training uses **fixed time and fixed azimuth**: no per-row truth alignment is
optimized in the loss or hidden in inference. This is a local adaptation tool,
not a reproduction of the paper's profiled distillation/training procedure.
If your targets have inconsistent physical reference phases or clocks, fix
their preparation before training rather than expecting the network to absorb
unrelated coordinate offsets.

The default loss is the mean of two normalized squared strain errors on
`[-1000,-160)` and `[-160,40]M`. Each window needs positive observed target
energy. Use `--window merger` when only `[-160,40]M` has usable support.
No accuracy outside the supervised window is claimed. There is no Psi4 loss
or synthetic teacher generation. Samples are drawn uniformly by physical
group, then uniformly among that group's rows; `--batch-size` is the number
of accumulated rows per AdamW update, processed one at a time to limit memory.

The optimizer is **new AdamW**, not the unpublished original optimizer.
The learning rate is constant; weight decay defaults to `1e-4`, gradient-norm
clipping to 1, and EMA decay to 0.99. EMA starts at the supplied model, not zero.
Validation evaluates the EMA with the same fixed-reference metric, averages
row L2 within each group and then averages groups. BEST includes the unchanged
parent at step zero, so an unsuccessful run can retain the parent as BEST.
`--patience 10` stops after ten validation checks without improvement.
Regularization and early stopping do not guarantee retention outside your data
domain; include suitable anchor systems in TRAIN and independent checks in
validation, particularly at high q and high spin.

## 3. Resume and load

Local outputs:

- `best_weights.pt`: minimum validation-group L2 EMA, directly loadable.
- `last_weights.pt`: last evaluated EMA, directly loadable.
- `state.pt`: raw parameters, EMA, AdamW and sampler state for resume.
- `metrics.jsonl`: per-check validation rows, aggregate scores and batch loss.
- `result.json`: current/completed run state and selection summary.

The output directory must be new for a fresh run. A resume keeps the optimizer,
EMA, sampler and early-stopping counters; repeat the same settings and increase
the **total** step budget:

```bash
python -m merger_transformer.finetune \
  --weights weights/weights.pt --manifest private_data/manifest.json \
  --output finetune_runs/private_nr --device cuda:0 \
  --steps 3000 --batch-size 4 --lr 1e-5 --eval-every 100 --resume
```

Local content fingerprints prevent accidental resume with changed data or
parent weights. Only the total step budget may change. After a crash, resume
starts at the last completed validation checkpoint; later unsaved updates are
repeated. Exact trajectory replay is tested on CPU on the same software stack;
cross-device/CUDA bitwise reproducibility is not promised.

```python
from merger_transformer import WaveformModel
adapted = WaveformModel.from_pretrained("finetune_runs/private_nr/best_weights.pt")
h, psi4 = adapted.waveforms(params, times)
```

Adapted exports have their own format and metadata. They remain compatible with
the current plotting, eager differentiation and optional frozen-inference API.
Use this package version to load them; historical v0.2.1 rejects adapted weights.
Original v0.2.1 weights still load unchanged. Further adaptation can start from
an adapted export, using a fresh output directory and new optimizer.

## Local-only data and smoke test

The trainer has no network calls. Private waveform arrays are not included in
inference exports, although fitted weights can still encode information about
their training data. Metrics and resume state are local research artifacts;
do not publish them unintentionally. `private_data/`, `finetune_runs/`, NPZ and
HDF5 files are ignored by this repository. No user data or adapted weights are
uploaded by these commands.

A fully synthetic end-to-end example is provided to exercise the machinery:

```bash
python package/examples/private_nr_smoke.py --weights weights/weights.pt \
  --output finetune_runs/synthetic_smoke
```

It generates model-derived perturbed targets, performs a few CPU updates, and
reloads the exported model. It is a software smoke test, not an NR accuracy or
generalization benchmark.
