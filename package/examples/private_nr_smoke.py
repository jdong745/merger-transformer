"""Synthetic end-to-end fine-tuning smoke; not an NR accuracy experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from merger_transformer import MODES, WaveformModel
from merger_transformer.finetune import DATA_FORMAT, MF_REF, REFERENCE_CONTRACT, parser, run


def prepare(weights, directory):
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = WaveformModel.from_pretrained(weights)
    times = np.array([-800., -600., -400., -200., -160., -120., -80., -40., 0., 40.])
    rows = []
    for i, q in enumerate((1.5, 2., 2.5)):
        params = np.array([q, .2, -.1, .3, -.05, .1, -.2])
        with torch.no_grad():
            target = 1.03 * model(params, times).numpy()
        name = f"synthetic_{i}"
        np.savez(directory / (name + ".npz"), params=params, times=times,
                 strain=target, mask=np.ones(target.shape, dtype=bool), modes=np.asarray(MODES))
        rows.append(dict(id=name, group=name, split="train" if i < 2 else "validation", file=name + ".npz"))
    manifest = dict(format=DATA_FORMAT, reference_contract=REFERENCE_CONTRACT,
                    time_coordinate="tau=(u-t0_curvature)/M", Mf_ref=MF_REF,
                    strain_convention="H=(R/M)*(h_plus-i*h_cross)", target_field="strain",
                    memory_convention="release_compatible_memory_subtracted", rows=rows)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New scratch directory")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = prepare(args.weights, args.output / "synthetic_data")
    settings = parser().parse_args(["--weights", str(args.weights), "--manifest", str(manifest),
                                   "--output", str(args.output / "run"), "--steps", "4",
                                   "--batch-size", "1", "--eval-every", "2", "--ema-decay", "0.5",
                                   "--replay-train-size", "8", "--replay-val-size", "8",
                                   "--device", args.device])
    result = run(settings)
    adapted = WaveformModel.from_pretrained(args.output / "run" / "last_weights.pt")
    x = torch.tensor([2., .2, -.1, .3, -.05, .1, -.2], dtype=torch.float64, requires_grad=True)
    h, psi4 = adapted.waveforms(x, [-20., 0., 20.])
    gradient = torch.autograd.grad(h.abs().square().sum(), x)[0]
    assert h.shape == psi4.shape == (21, 3)
    assert all(bool(torch.isfinite(v).all()) for v in (h, psi4, gradient))
    print(json.dumps({"synthetic_smoke": "PASS", "training": result}, indent=2))


if __name__ == "__main__":
    main()
