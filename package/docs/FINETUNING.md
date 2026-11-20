# Fine-tuning with private NR waveforms

The optional trainer (`merger-transformer-finetune`) adapts the **released v0.5.0 extended-window STEP064000 EMA model** to locally supplied strain targets, with frozen original-model replay to constrain forgetting outside your NR sample. Full-network adaptation is the default. Everything runs locally; no NR data or adapted weights are uploaded.

```bash
python -m pip install './package[finetune]'
```

## 1. Prepare local targets

Each simulation is stored as one NPZ file containing:

| Array | Shape / type | Meaning |
| --- | --- | --- |
| `params` | `[7]`, real | `(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)` |
| `times` | `[T]`, real | Strictly increasing dimensionless `tau`, within `[-1620, 160]` |
| `strain` | `[21, T]`, complex | Distance-scaled `H_lm`, ordered as `merger_transformer.MODES` |
| `modes` | `[21, 2]`, integer | Explicit ordered `(ell, m)` list matching `MODES` |
| `mask` | `[T]` or `[21, T]`, boolean | `True` only for valid measured samples/modes |

Conventions must match the base release:
- `q = m1/m2 > 0`; spins in `(n, Lhat × n, Lhat)` at `M f_ref = 0.01182117827433904`.
- `H = (R/M) (h_plus - i h_cross)` with BMS energy-flux memory subtracted.
- Time origin `tau = (u - t0_curvature)/M` defined from the curvature-power moment.

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

Validate before training:

```bash
python -m merger_transformer.finetune --weights weights/weights.pt \
  --manifest private_data/manifest.json --check-data
```

## 2. Run adaptation

```bash
merger-transformer-finetune \
  --weights weights/weights.pt --manifest private_data/manifest.json \
  --output finetune_runs/private_nr --device cuda:0 \
  --steps 2000 --batch-size 1 --lr 1e-6 --eval-every 100
```

- `--window balanced` (default) weights `[-1500, -160)M` and `[-160, 100]M` normalized squared strain errors equally; `--window merger` uses `[-160, 100]M` only.
- `--trainable all` (default) updates all `2,564,276` parameters; `--trainable head` updates only output/mode heads. Both preserve the four symmetry branches and exact 518-knot spline basis.
- Nine frozen original-model replay waveforms per private NR row (`replay_train_size=512`, `replay_val_size=128` across `[-1500, 100]M` in four `q` strata) constrain forgetting. Optional `--max-replay-l2 VALUE` enforces a per-stratum retention bound when selecting `best_weights.pt`.

## 3. Resume and load

Resume by repeating the command with `--resume` and a larger `--steps` (`STATE_FORMAT = merger-transformer-finetune-state-v3`):

```bash
merger-transformer-finetune \
  --weights weights/weights.pt --manifest private_data/manifest.json \
  --output finetune_runs/private_nr --device cuda:0 \
  --steps 3000 --batch-size 1 --lr 1e-6 --eval-every 100 --resume
```

Load adapted exports directly:

```python
from merger_transformer import WaveformModel
adapted = WaveformModel.from_pretrained("finetune_runs/private_nr/best_weights.pt")
h, psi4 = adapted.waveforms(params, times)
```

For synthetic end-to-end verification:

```bash
python package/examples/private_nr_smoke.py --weights weights/weights.pt --output finetune_runs/smoke
```
