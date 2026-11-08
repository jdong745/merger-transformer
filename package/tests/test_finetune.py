"""Private-data guards and actual bounded optimization/export/resume controls."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from merger_transformer import MODES, WaveformModel
from merger_transformer.finetune import load_dataset, loss_weights, parser, run
from merger_transformer._replay import STRATA, sample_inputs


@pytest.fixture
def manifest(tmp_path, weights_path):
    example = Path(__file__).resolve().parents[1] / "examples/private_nr_smoke.py"
    spec = importlib.util.spec_from_file_location("private_smoke", example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare(weights_path, tmp_path / "data")


def settings(weights, manifest, output, *extra):
    return parser().parse_args(["--weights", str(weights), "--manifest", str(manifest),
                               "--output", str(output), "--steps", "2", "--batch-size", "1",
                               "--eval-every", "2", "--ema-decay", "0.5", "--trainable", "head",
                               "--replay-per-nr", "1", "--replay-train-size", "4",
                               "--replay-val-size", "4", *extra])


def test_masked_quadrature():
    t = np.array([-500., -400., -300., -200., -160., -80., 0., 40.])
    h = np.ones((21, len(t)), dtype=complex)
    mask = np.ones(h.shape, dtype=bool)
    mask[:, 1] = False
    w = loss_weights(t, h, mask)
    assert (w[:, 0] == 0).all()  # isolated point: no quadrature across masked gap
    assert np.isclose(np.sum(w * abs(.03 * h)**2), .03**2)


def test_data_validation(manifest):
    rows, fingerprint = load_dataset(manifest)
    assert len(rows) == 3 and len(fingerprint) == 64
    content = json.loads(manifest.read_text())
    content["rows"][-1]["group"] = content["rows"][0]["group"]
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="crosses"):
        load_dataset(manifest)


def test_modes_and_nan_mask(manifest):
    source = manifest.parent / "synthetic_0.npz"
    with np.load(source) as data:
        arrays = dict(data)
    arrays["strain"][0, 0] = np.nan
    np.savez(source, **arrays)
    with pytest.raises(ValueError, match="nonfinite observed"):
        load_dataset(manifest)
    arrays["mask"][0, 0] = False
    np.savez(source, **arrays)
    assert np.isfinite(load_dataset(manifest)[0][0].strain).all()
    arrays["modes"] = arrays["modes"][::-1]
    np.savez(source, **arrays)
    with pytest.raises(ValueError, match="ordered"):
        load_dataset(manifest)


def test_head_training_export_and_exact_resume(manifest, weights_path, tmp_path):
    short, full = tmp_path / "short", tmp_path / "full"
    result = run(settings(weights_path, manifest, short))
    assert result["step"] == 2
    parent = torch.load(weights_path, weights_only=True)["state_dict"]
    state = torch.load(short / "state.pt", weights_only=True)
    changed = [k for k, v in state["raw"].items() if not torch.equal(v, parent[k])]
    assert changed and all(k.startswith(("output.", "mode_head.")) for k in changed)
    assert result["best_validation_fixed_group_L2"] <= .03 / 1.03 + 1e-12
    # Exercise migration of 0.3.1 states, which lack explicit path metadata.
    state["settings"].pop("teacher_device")
    state["settings"].pop("numerical_path")
    torch.save(state, short / "state.pt")
    run(settings(weights_path, manifest, short, "--resume", "--steps", "4"))
    run(settings(weights_path, manifest, full, "--steps", "4"))
    resumed = torch.load(short / "state.pt", weights_only=True)
    continuous = torch.load(full / "state.pt", weights_only=True)
    for kind in ("raw", "ema"):
        assert all(torch.equal(v, continuous[kind][k]) for k, v in resumed[kind].items())
    assert torch.equal(resumed["sampler"], continuous["sampler"])
    assert resumed["best"] == continuous["best"]
    cache = torch.load(short / "replay.pt", weights_only=True)
    teacher = WaveformModel.from_pretrained(weights_path)
    with torch.no_grad():
        r = cache["train"][0]
        torch.testing.assert_close(teacher(r["params"], r["times"]), r["strain"], rtol=0, atol=0)
    metrics = [json.loads(line) for line in (short / "metrics.jsonl").read_text().splitlines()]
    for record in metrics[1:]:
        assert set(record["retention_by_q"]) == set(STRATA)
        assert record["train_batch_loss"] == pytest.approx(record["train_private_loss"] + record["train_replay_loss"])
    exported = torch.load(short / "last_weights.pt", weights_only=True)
    assert "synthetic_" not in str(exported["metadata"])
    model = WaveformModel.from_pretrained(short / "last_weights.pt")
    x = torch.tensor([2., .2, -.1, .3, -.05, .1, -.2], dtype=torch.float64, requires_grad=True)
    h, p = model.waveforms(x, [-20., 0., 20.])
    assert h.shape == p.shape == (21, 3) and torch.isfinite(p).all()
    assert torch.isfinite(torch.autograd.grad(h.abs().square().sum(), x)[0]).all()
    turn = torch.tensor([-1., -1., 1.], dtype=torch.float64)
    exchanged = torch.cat((1/x[:1], turn*x[4:], turn*x[1:4]))
    reflected = torch.cat((x[:1], turn*x[1:4], turn*x[4:]))
    with torch.no_grad():
        c, ce, cp = model.coefficients(torch.stack((x, exchanged, reflected)))
    for i, (ell, m) in enumerate(MODES):
        torch.testing.assert_close(ce[i], (-1)**m*c[i], rtol=2e-11, atol=3e-14)
        torch.testing.assert_close(cp[i], (-1)**ell*c[MODES.index((ell, -m))].conj(), rtol=2e-11, atol=3e-14)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="identical data"):
        run(settings(weights_path, manifest, short, "--resume", "--steps", "6"))


def test_all_parameters_float32_export(manifest, weights_path, tmp_path):
    out = tmp_path / "float32"
    run(settings(weights_path, manifest, out, "--trainable", "all", "--dtype", "float32", "--steps", "1"))
    raw = torch.load(out / "state.pt", weights_only=True)["raw"]
    parent = torch.load(weights_path, weights_only=True)["state_dict"]
    assert any(not torch.equal(v, parent[k].float()) for k, v in raw.items()
               if not k.startswith(("output.", "mode_head.", "basis.")))
    model = WaveformModel.from_pretrained(out / "last_weights.pt")
    assert torch.equal(model.network.basis.knots, parent["basis.knots"])
    assert torch.isfinite(model([2., .2, -.1, .3, -.05, .1, -.2], [0.])).all()


def test_replay_distribution_and_defaults():
    train = sample_inputs(64, 123)
    val = sample_inputs(64, 456)
    assert len({tuple(x) for _, x in train + val}) == 128
    assert all(sum(name == s for name, _ in train) == 16 for s in STRATA)
    for s, x in train:
        assert max(np.linalg.norm(x[1:4]), np.linalg.norm(x[4:])) <= .9
        if s == STRATA[3]:
            assert 20 < x[0] < 1000 and -.5 <= x[3] <= .8
            assert (x[[1, 2, 4, 5, 6]] == 0).all()
    defaults = parser().parse_args(["--weights", "unused.pt", "--manifest", "unused.json"])
    assert defaults.trainable == "all" and defaults.replay_per_nr == 9
    assert defaults.lr == 1e-6 and defaults.replay_weight == 1.
    assert defaults.max_replay_l2 is None


def test_retention_bound_keeps_parent(manifest, weights_path, tmp_path):
    out = tmp_path / "guard"
    result = run(settings(weights_path, manifest, out, "--steps", "1", "--max-replay-l2", "1e-12"))
    assert result["best_step"] == 0
    metrics = [json.loads(s) for s in (out / "metrics.jsonl").read_text().splitlines()]
    assert metrics[0]["retention_eligible"] and not metrics[-1]["retention_eligible"]
    best = torch.load(out / "best_weights.pt", weights_only=True)["state_dict"]
    parent = torch.load(weights_path, weights_only=True)["state_dict"]
    assert all(torch.equal(v, parent[k]) for k, v in best.items())
    adapted = out / "last_weights.pt"
    with pytest.raises(ValueError, match="teacher must be"):
        run(settings(adapted, manifest, tmp_path / "bad_teacher"))
    # Original released model remains usable as the explicit retention teacher.
    run(settings(adapted, manifest, tmp_path / "adapt_again", "--teacher-weights", str(weights_path), "--steps", "1"))
