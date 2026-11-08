"""Inference, symmetry and derivative checks; no training or NR data required."""
import numpy as np
import pytest
import torch
from scipy.interpolate import BSpline

from merger_transformer import MODES, WaveformModel


def test_export_and_random_stream(weights_path):
    payload = torch.load(weights_path, weights_only=True, map_location="cpu")
    assert set(payload) == {"format", "model_config", "state_dict", "metadata"}
    assert payload["metadata"]["weight_kind"] == "EMA"
    assert len(payload["state_dict"]) == 105
    assert all(v.dtype == torch.float64 for v in payload["state_dict"].values())
    before = torch.get_rng_state().clone()
    net = WaveformModel.from_pretrained(weights_path)
    assert torch.equal(before, torch.get_rng_state())
    assert sum(p.numel() for p in net.parameters()) == 1974036
    assert not any(p.requires_grad for p in net.parameters())


def test_spline_against_scipy(model, parameters):
    with torch.no_grad():
        coefficients = model.coefficients(parameters)
    knots = model.network.basis.knots.cpu().numpy()
    times = np.unique(np.r_[np.linspace(-1100, 80, 311), knots])
    independent = BSpline(knots, coefficients.cpu().numpy(), 5, axis=-1, extrapolate=False)
    for derivative in (0, 1, 2):
        actual = model.evaluate_coefficients(coefficients, times, derivative=derivative).cpu().numpy()
        np.testing.assert_allclose(actual, independent(times, nu=derivative), rtol=3e-11, atol=2e-14)


def test_time_autograd_matches_curvature(model, parameters):
    times = torch.tensor([-1100., -305.25, -20.25, 0., 80.], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        c = model.coefficients(parameters)
    h = model.evaluate_coefficients(c, times)
    expected1 = model.evaluate_coefficients(c, times, derivative=1)
    expected2 = model.evaluate_coefficients(c, times, derivative=2)
    for component in ("real", "imag"):
        dt = torch.autograd.grad(getattr(h[4], component).sum(), times, create_graph=True, retain_graph=True)[0]
        d2t = torch.autograd.grad(dt.sum(), times, retain_graph=True)[0]
        torch.testing.assert_close(dt, getattr(expected1[4], component), rtol=2e-11, atol=2e-14)
        torch.testing.assert_close(d2t, getattr(expected2[4], component), rtol=2e-11, atol=2e-14)
    torch.testing.assert_close(model.psi4(parameters, times), -expected2)


def test_parameter_gradient_and_second_derivative(model, parameters):
    x = parameters.requires_grad_(True)
    t = torch.tensor([-20.25], dtype=torch.float64)
    value = model(x, t)[4, 0].real
    gradient = torch.autograd.grad(value, x, create_graph=True)[0]
    direction = torch.tensor([.4, -.2, .3, .1, -.1, .2, .1], dtype=torch.float64)
    hessian_direction = torch.autograd.grad((gradient * direction).sum(), x)[0]
    assert torch.isfinite(hessian_direction).all()
    epsilon = 1e-6
    probes = torch.eye(7, dtype=torch.float64) * epsilon
    with torch.no_grad():
        plus = model(x.detach()[None, :] + probes, t)[:, 4, 0].real
        minus = model(x.detach()[None, :] - probes, t)[:, 4, 0].real
    torch.testing.assert_close(gradient, (plus - minus) / (2 * epsilon), rtol=2e-5, atol=2e-7)
    grads = []
    for sign in (1., -1.):
        shifted = (x.detach() + sign * epsilon * direction).requires_grad_(True)
        scalar = model(shifted, t)[4, 0].real
        grads.append(torch.autograd.grad(scalar, shifted)[0])
    torch.testing.assert_close(hessian_direction, (grads[0] - grads[1]) / (2 * epsilon), rtol=2e-5, atol=2e-6)


def test_exchange_reflection_and_aligned_odd_modes(model, parameters):
    turn = torch.tensor([-1., -1., 1.], dtype=torch.float64)
    exchanged = torch.cat((1 / parameters[:1], turn * parameters[4:], turn * parameters[1:4]))
    reflected = torch.cat((parameters[:1], turn * parameters[1:4], turn * parameters[4:]))
    aligned_equal = torch.tensor([1., 0., 0., .3, 0., 0., .3], dtype=torch.float64)
    with torch.no_grad():
        c, ce, cp, ca = model.coefficients(torch.stack((parameters, exchanged, reflected, aligned_equal)))
    for i, (ell, m) in enumerate(MODES):
        opposite = MODES.index((ell, -m))
        torch.testing.assert_close(ce[i], (-1) ** m * c[i], rtol=2e-11, atol=3e-14)
        torch.testing.assert_close(cp[i], (-1) ** ell * c[opposite].conj(), rtol=2e-11, atol=3e-14)
        torch.testing.assert_close(ca[opposite], (-1) ** ell * ca[i].conj(), rtol=2e-11, atol=3e-14)
        if m % 2:
            assert ca[i].abs().max().item() < 3e-14


def test_shapes_dictionary_and_microbatches(model, parameters, weights_path):
    times = torch.tensor([-1000., -160., -20.25, 40.], dtype=torch.float64)
    other = parameters.clone()
    other[0] = 8.
    inputs = torch.stack((parameters, other))
    single_batch = WaveformModel.from_pretrained(weights_path, microbatch_size=1)
    with torch.no_grad():
        batched = model(inputs, times)
        individual = single_batch(inputs, times)
        single = model(parameters, times)
        h, p = model.waveforms(parameters, times)
        by_mode = model.mode_dict(parameters, times)
    assert batched.shape == (2, 21, 4)
    assert single.shape == (21, 4)
    torch.testing.assert_close(individual, batched, rtol=2e-11, atol=2e-14)
    torch.testing.assert_close(single, batched[0], rtol=2e-11, atol=2e-14)
    torch.testing.assert_close(h, single)
    torch.testing.assert_close(p, model.psi4(parameters, times))
    assert tuple(by_mode) == MODES
    torch.testing.assert_close(by_mode[(2, 2)], single[4])


@pytest.mark.parametrize("bad", [[0., 0, 0, 0, 0, 0, 0], [-1., 0, 0, 0, 0, 0, 0],
                                 [2., 1, 1, 0, 0, 0, 0], [2., float("nan"), 0, 0, 0, 0, 0], [1., 2.]])
def test_parameter_guards(model, bad):
    with pytest.raises(ValueError):
        model(bad, [0.])


@pytest.mark.parametrize("bad", [[], [[0.]], [-1100.0001], [80.0001], [float("nan")]])
def test_time_guards(model, parameters, bad):
    with pytest.raises(ValueError):
        model(parameters, bad)


def test_reference_frame_smooth_gradient_at_equal_mass(model):
    x = torch.tensor([1., 0., 0., .2, 0., 0., -.1], dtype=torch.float64, requires_grad=True)
    h = model(x, [-20.25])
    grad = torch.autograd.grad(h[4, 0].real, x)[0]
    assert torch.isfinite(grad).all()
    eps = 1e-6
    probe = torch.zeros_like(x)
    probe[0] = eps
    with torch.no_grad():
        finite = (model(x + probe, [-20.25])[4, 0].real - model(x - probe, [-20.25])[4, 0].real) / (2 * eps)
    torch.testing.assert_close(grad[0], finite, rtol=2e-5, atol=2e-7)


def test_single_precision_optional(weights_path, parameters):
    model = WaveformModel.from_pretrained(weights_path, dtype=torch.float32)
    with torch.no_grad():
        h, p = model.waveforms(parameters.float(), [0., 40.])
    assert h.dtype == torch.complex64 and p.dtype == torch.complex64
    assert torch.isfinite(h).all() and torch.isfinite(p).all()


def test_python_lists_preserve_double_precision(model):
    values = [2.123456789123, .123456789123, -.234567891234, .312345678912,
              -.051234567891, .112345678912, -.212345678912]
    times = [-20.123456789123]
    x = torch.tensor(values, dtype=torch.float64)
    t = torch.tensor(times, dtype=torch.float64)
    with torch.no_grad():
        actual = model(values, times)
        expected = model(x, t)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rejects_other_checkpoint_identity(weights_path, tmp_path):
    payload = torch.load(weights_path, weights_only=True, map_location="cpu")
    payload["metadata"]["selected_step"] = 40000
    wrong = tmp_path / "wrong_identity.pt"
    torch.save(payload, wrong)
    with pytest.raises(ValueError):
        WaveformModel.from_pretrained(wrong)


def test_rejects_inconsistent_knots(weights_path, tmp_path):
    payload = torch.load(weights_path, weights_only=True, map_location="cpu")
    payload["state_dict"]["basis.knots"][10] += .01
    wrong = tmp_path / "wrong_knots.pt"
    torch.save(payload, wrong)
    with pytest.raises(ValueError):
        WaveformModel.from_pretrained(wrong)


@pytest.mark.parametrize("field,value", [("model", "v7_width192_recovery"), ("continuation_arm", "decay"), ("continuation_learning_rate", 1e-3)])
def test_rejects_wrong_continuation(weights_path, tmp_path, field, value):
    payload = torch.load(weights_path, weights_only=True, map_location="cpu")
    payload["metadata"][field] = value
    wrong = tmp_path / "wrong_continuation.pt"
    torch.save(payload, wrong)
    with pytest.raises(ValueError):
        WaveformModel.from_pretrained(wrong)
