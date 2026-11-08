"""Deterministic synthetic retention panels; no NR data or fitted input labels."""
import math
import numpy as np
import torch

STRATA = ("q1_4", "q4_8", "q8_20", "aligned_q20_1000")
REPLAY_VERSION = 1


def sample_inputs(count, seed):
    """Equal counts by q stratum (up to rounding), log-uniform q.

    First three strata: 1/4 aligned, otherwise isotropic spin directions;
    spin magnitudes uniform [0,.8] below q4, [0,.9] above. Last stratum:
    aligned primary spin [-.5,.8], zero secondary spin, like the published
    perturbative teacher domain. This preserves predictions, not NR truth.
    """
    rng = np.random.default_rng(seed)
    rows = []
    bounds = ((1., 4.), (4., 8.), (8., 20.), (20., 1000.))
    for i in range(count):
        stratum = i % 4
        lo, hi = bounds[stratum]
        q = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        x = np.zeros(7)
        x[0] = q
        if stratum == 3:
            x[3] = rng.uniform(-.5, .8)
        else:
            bound = .8 if stratum == 0 else .9
            if rng.random() < .25:
                x[3], x[6] = rng.uniform(-bound, bound, 2)
            else:
                directions = rng.normal(size=(2, 3))
                directions /= np.linalg.norm(directions, axis=1)[:, None]
                x[1:] = (directions * rng.uniform(0, bound, (2, 1))).reshape(6)
        rows.append((STRATA[stratum], x))
    return rows


@torch.no_grad()
def build_panel(teacher, count, seed, split):
    from .finetune import Row, loss_weights
    # Full published scoring core, 1M spacing, all 21 modes, fixed frame/time.
    times = np.arange(-1000., 41.)
    rows = []
    for i, (stratum, x) in enumerate(sample_inputs(count, seed)):
        h = teacher(x, times).cpu().numpy().astype(np.complex128)
        if not np.isfinite(h).all():
            raise FloatingPointError(f"Nonfinite frozen replay target: {stratum}, {i}")
        weights = loss_weights(times, h, np.ones(h.shape, dtype=bool))
        identity = f"{stratum}:{i}"
        rows.append(Row(identity, identity, split, x, times, h, weights))
    return rows


def panel_tensors(rows):
    return [{"id": r.id, "group": r.group, "split": r.split,
             **{name: torch.from_numpy(getattr(r, name))
                for name in ("params", "times", "strain", "weights")}} for r in rows]


def panel_rows(records):
    from .finetune import Row
    return [Row(r["id"], r["group"], r["split"],
                *(r[name].cpu().numpy() for name in ("params", "times", "strain", "weights")))
            for r in records]


def retention_summary(records):
    means = {}
    for name in STRATA:
        values = [r["fixed_L2"] for r in records if r["group"].split(":")[0] == name]
        if not values:
            raise ValueError(f"Missing retention validation stratum: {name}")
        means[name] = sum(values) / len(values)
    return means
