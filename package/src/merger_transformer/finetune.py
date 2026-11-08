"""Local private-NR adaptation of the released model; no downloads or uploads.

This deliberately small eager trainer uses fixed physical time and azimuth.
It is not the catalogue-preparation or distillation pipeline from the paper.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import os

import numpy as np
import torch

from .api import WaveformModel, ADAPTED_WEIGHT_FORMAT
from ._network import MODES, REFERENCE_CONTRACT
from ._basis import QuinticBasis

DATA_FORMAT = "merger-transformer-private-nr-v1"
STATE_FORMAT = "merger-transformer-finetune-state-v1"
MF_REF = 0.01182117827433904


@dataclass
class Row:
    id: str
    group: str
    split: str
    params: np.ndarray
    times: np.ndarray
    strain: np.ndarray
    weights: np.ndarray


def loss_weights(times, strain, mask, window="balanced"):
    """Trapezoidal quadrature, independently normalized in each active window.

    No integration across gaps in a mode's mask. Equal-weight normalized early
    [-1000,-160) and merger [-160,40] errors, or merger only. Unknown modes do
    not contribute to either the numerator or the target-energy denominator.
    """
    pieces = [(-160., 40.)] if window == "merger" else [(-1000., -160.), (-160., 40.)]
    weights = np.zeros(strain.shape, dtype=np.float64)
    for index, (lo, hi) in enumerate(pieces):
        inside = (times >= lo) & ((times < hi) if hi == -160 else (times <= hi))
        valid = mask & inside[None, :]
        adjacent = valid[:, :-1] & valid[:, 1:]
        half = adjacent * (np.diff(times)[None, :] / 2)
        quadrature = np.zeros_like(weights)
        quadrature[:, :-1] += half
        quadrature[:, 1:] += half
        energy = float(np.sum(quadrature * abs(strain)**2))
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError(f"No positive observed strain energy in window {index}: [{lo},{hi}]")
        weights += quadrature / energy / len(pieces)
    return weights


def load_dataset(path: Path, window="balanced"):
    """Read only explicitly supplied local files; validate before training."""
    content = path.read_bytes()
    manifest = json.loads(content)
    expected = {
        "format": DATA_FORMAT,
        "reference_contract": REFERENCE_CONTRACT,
        "time_coordinate": "tau=(u-t0_curvature)/M",
        "strain_convention": "H=(R/M)*(h_plus-i*h_cross)",
        "target_field": "strain",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Manifest {key} must be {value!r}")
    if not math.isclose(float(manifest.get("Mf_ref", -1)), MF_REF, rel_tol=0, abs_tol=1e-14):
        raise ValueError("Input spins must use the released model's Mf_ref")
    if manifest.get("memory_convention") != "release_compatible_memory_subtracted":
        raise ValueError("Declare release-compatible oscillatory/memory-subtracted strain targets")
    entries = manifest.get("rows")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest needs a nonempty rows list")
    digest = hashlib.sha256(content)
    seen, group_roles, rows = set(), {}, []
    for item in entries:
        sid, group, split = (item.get(k) for k in ("id", "group", "split"))
        if not all(isinstance(v, str) and v for v in (sid, group, split)):
            raise ValueError("Every row needs nonempty string id/group/split")
        if sid in seen or split not in ("train", "validation"):
            raise ValueError(f"Duplicate ID or invalid split: {sid}")
        seen.add(sid)
        if group in group_roles and group_roles[group] != split:
            raise ValueError(f"Physical group crosses train/validation: {group}")
        group_roles[group] = split
        source = path.parent / item["file"]
        raw = source.read_bytes()
        digest.update(raw)  # local resume guard only; no public checksum inventory
        with np.load(source, allow_pickle=False) as data:
            required = {"params", "times", "strain", "modes", "mask"}
            if set(data.files) != required:
                raise ValueError(f"{sid}: NPZ must contain exactly {sorted(required)}")
            x, t, h, modes, mask = (data[k] for k in ("params", "times", "strain", "modes", "mask"))
        if x.shape != (7,) or not np.isrealobj(x) or not np.isfinite(x).all() or x[0] <= 0:
            raise ValueError(f"{sid}: invalid physical parameters")
        if max(np.linalg.norm(x[1:4]), np.linalg.norm(x[4:])) > 1 + 1e-12:
            raise ValueError(f"{sid}: spin magnitude exceeds one")
        if t.ndim != 1 or len(t) < 3 or not np.isrealobj(t) or not np.isfinite(t).all():
            raise ValueError(f"{sid}: invalid time array")
        if np.any(np.diff(t) <= 0) or t[0] < -1100 or t[-1] > 80:
            raise ValueError(f"{sid}: times must increase within [-1100,80]M")
        if not np.array_equal(modes, np.asarray(MODES)):
            raise ValueError(f"{sid}: use the complete ordered 21 signed MODES axis")
        if h.shape != (21, len(t)) or not np.iscomplexobj(h):
            raise ValueError(f"{sid}: strain must be complex [21,T]")
        if mask.dtype != np.bool_ or mask.shape not in ((len(t),), h.shape):
            raise ValueError(f"{sid}: mask must be boolean [T] or [21,T]")
        mask = np.broadcast_to(mask, h.shape)
        if not np.isfinite(h[mask]).all():
            raise ValueError(f"{sid}: nonfinite observed strain")
        h = np.where(mask, h, 0).astype(np.complex128)
        w = loss_weights(t, h, mask, window)
        rows.append(Row(sid, group, split, x.astype(np.float64), t.astype(np.float64), h, w))
    if {r.split for r in rows} != {"train", "validation"}:
        raise ValueError("Provide at least one TRAIN group and one independent validation group")
    return rows, digest.hexdigest()


def row_loss(model, row):
    target = torch.as_tensor(row.strain, device=model.device,
                             dtype=torch.complex128 if model.dtype == torch.float64 else torch.complex64)
    weight = torch.as_tensor(row.weights, device=model.device, dtype=model.dtype)
    predicted = model(row.params, row.times)
    residual = predicted - target
    return ((residual.real.square() + residual.imag.square()) * weight).sum()


@torch.no_grad()
def validate(model, rows, ema):
    raw = {k: v.detach().clone() for k, v in model.network.state_dict().items()}
    model.network.load_state_dict(ema, strict=True)
    groups, records = defaultdict(list), []
    try:
        for row in rows:
            value = float(row_loss(model, row).sqrt())
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite validation loss: {row.id}")
            groups[row.group].append(value)
            records.append({"id": row.id, "group": row.group, "fixed_L2": value})
        score = sum(sum(v) / len(v) for v in groups.values()) / len(groups)
    finally:
        model.network.load_state_dict(raw, strict=True)
    return score, records


def atomic_save(value, path):
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def export_weights(path, model, ema, parent, step, score, settings):
    base = parent.get("base_product", parent)
    metadata = {
        "model": "private_nr_finetune", "weight_kind": "EMA",
        "base_product": base,
        "adaptation": {"steps": step, "validation_fixed_group_L2": score,
                       "trainable": settings["trainable"], "window": settings["window"],
                       "alignment": "fixed_time_and_azimuth", "published_product": False},
    }
    # Private waveform arrays, IDs, file paths and optimizer states are excluded.
    state = {k: v.detach().cpu().double() for k, v in ema.items()}
    # Restore the immutable binary64 grid after optional float32 training.
    # Parameter precision is not recovered by exporting to float64.
    state["basis.knots"] = QuinticBasis().knots
    metadata["adaptation"]["training_dtype"] = settings["dtype"]
    atomic_save({"format": ADAPTED_WEIGHT_FORMAT, "model_config": model.network.config_dict(),
                 "state_dict": state,
                 "metadata": metadata}, path)


def run(args):
    for name in ("steps", "batch_size", "eval_every", "threads", "patience"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("lr", "clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 <= args.ema_decay < 1 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("Invalid EMA decay or weight decay")
    rows, fingerprint = load_dataset(args.manifest, args.window)
    if args.check_data:
        summary = {split: {"rows": sum(r.split == split for r in rows),
                           "groups": len({r.group for r in rows if r.split == split})}
                   for split in ("train", "validation")}
        print(json.dumps(summary, indent=2))
        return summary
    torch.set_num_threads(args.threads)
    model = WaveformModel.from_pretrained(args.weights, device=args.device,
                                          dtype=getattr(torch, args.dtype), microbatch_size=1)
    # Train the same shared network; symmetry projection and spline basis stay intact.
    model.eval()
    for name, parameter in model.network.named_parameters():
        parameter.requires_grad_(args.trainable == "all" or name.startswith(("output.", "mode_head.")))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    ema = {k: v.detach().clone() for k, v in model.network.state_dict().items()}
    sampler = torch.Generator(device="cpu").manual_seed(args.seed)
    train, validation = [r for r in rows if r.split == "train"], [r for r in rows if r.split == "validation"]
    groups = defaultdict(list)
    for row in train:
        groups[row.group].append(row)
    group_names = sorted(groups)
    settings = {k: getattr(args, k) for k in ("trainable", "window", "dtype", "device", "threads",
                 "batch_size", "lr", "weight_decay", "clip", "ema_decay", "seed", "eval_every", "patience")}
    out = args.output
    initial_bytes = args.weights.read_bytes()
    parent_digest = hashlib.sha256(initial_bytes).hexdigest()
    step, best, best_step, bad = 0, float("inf"), 0, 0
    if args.resume:
        state = torch.load(out / "state.pt", map_location=model.device, weights_only=True)
        if (state.get("format") != STATE_FORMAT or state["settings"] != settings
                or state["data_fingerprint"] != fingerprint or state["parent_fingerprint"] != parent_digest):
            raise ValueError("Resume requires identical data, parent weights and settings; only --steps may increase")
        model.network.load_state_dict(state["raw"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        ema = state["ema"]
        sampler.set_state(state["sampler"].cpu())
        step, best, best_step, bad = (state[k] for k in ("step", "best", "best_step", "bad"))
        if args.steps <= step:
            raise ValueError("--steps must exceed the saved update count")
    else:
        out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps({"status": "running", "resumed_from_step": step}) + "\n")

    def checkpoint(current, training_loss=None):
        nonlocal best, best_step, bad
        score, records = validate(model, validation, ema)
        improved = score < best
        if improved:
            best, best_step, bad = score, current, 0
            export_weights(out / "best_weights.pt", model, ema, model.metadata, current, score, settings)
        elif current:
            bad += 1
        export_weights(out / "last_weights.pt", model, ema, model.metadata, current, score, settings)
        record = {"step": current, "train_batch_loss": training_loss, "validation_fixed_group_L2": score,
                  "best_step": best_step, "best_validation_fixed_group_L2": best,
                  "validation_rows": records}
        # Everything in this output directory is local and may contain private IDs.
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        atomic_save({"format": STATE_FORMAT, "step": current, "raw": model.network.state_dict(),
                     "ema": ema, "optimizer": optimizer.state_dict(), "sampler": sampler.get_state(),
                     "settings": settings, "data_fingerprint": fingerprint, "parent_fingerprint": parent_digest,
                     "best": best, "best_step": best_step, "bad": bad}, out / "state.pt")
        print(json.dumps({k: v for k, v in record.items() if k != "validation_rows"}), flush=True)

    if not args.resume:
        checkpoint(0)  # the unchanged parent is eligible as BEST
    reason = "step_budget"
    for step in range(step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        # Uniform physical groups, then uniform simulations within each group.
        # Per-row backward bounds activation memory even for irregular grids.
        for _ in range(args.batch_size):
            group = group_names[int(torch.randint(len(group_names), (), generator=sampler))]
            choices = groups[group]
            row = choices[int(torch.randint(len(choices), (), generator=sampler))]
            loss = row_loss(model, row) / args.batch_size
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss; last saved state remains available")
            total += float(loss.detach())
            loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.clip, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for name, value in model.network.state_dict().items():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Nonfinite update; last saved state remains available")
                ema[name].lerp_(value, 1 - args.ema_decay)
        if step % args.eval_every == 0 or step == args.steps:
            checkpoint(step, total)
            if bad >= args.patience:
                reason = "validation_patience"
                break
    result = {"status": "complete", "reason": reason, "step": step,
              "best_step": best_step, "best_validation_fixed_group_L2": best,
              "trainable_parameters": sum(p.numel() for p in parameters),
              "exports": ["best_weights.pt", "last_weights.pt"], "alignment": "fixed_time_and_azimuth"}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("finetune_runs/private_nr"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--trainable", choices=("head", "all"), default="head")
    p.add_argument("--window", choices=("balanced", "merger"), default="balanced")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--patience", type=int, default=10, help="Validation checks without improvement")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.)
    p.add_argument("--ema-decay", type=float, default=.99)
    p.add_argument("--seed", type=int, default=202)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-data", action="store_true")
    return p


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
