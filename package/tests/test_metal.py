"""Real-channel arithmetic controls; actual Metal tests skip without Apple hardware."""
import numpy as np
import pytest
import torch
from merger_transformer import MODES, WaveformModel
from merger_transformer._network import attend_unfused, transform_inputs, transform_modes
from merger_transformer.finetune import Row, loss_weights, row_loss, row_loss_ri


def as_complex(pair):
    return torch.complex(pair[..., 0], pair[..., 1])


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_real_channel_waveforms_and_projection(weights_path, dtype):
    model = WaveformModel.from_pretrained(weights_path, dtype=dtype, microbatch_size=1)
    x = torch.tensor([[2., .2, -.1, .3, -.05, .1, -.2],
                      [1., 0., 0., .3, 0., 0., .3],
                      [8., .7, .1, .1, 0., .1, 0.],
                      [1000., 0., 0., .8, 0., 0., 0.]], dtype=dtype)
    t = torch.linspace(-1100, 80, 237, dtype=dtype)
    tol = 2e-6 if dtype == torch.float32 else 2e-14
    with torch.no_grad():
        c = model.coefficients(x)
        model.network.mode_head.chunk = 256  # Metal memory chunking
        ri = model.coefficients_ri(x)
        torch.testing.assert_close(as_complex(ri), c, rtol=tol, atol=tol * .01)
        for order in range(3):
            torch.testing.assert_close(as_complex(model.evaluate_coefficients_ri(ri, t, derivative=order)),
                                       model.evaluate_coefficients(c, t, derivative=order), rtol=tol, atol=tol*.01)
        for g in ('E', 'P', 'EP'):
            cg = as_complex(model.coefficients_ri(transform_inputs(x, g)))
            torch.testing.assert_close(cg, transform_modes(c, g), rtol=tol*10, atol=tol*.1)
        odd = [i for i, (_, m) in enumerate(MODES) if m % 2]
        assert ri[1, odd].abs().max() < tol


def test_real_channel_gradient_and_update(weights_path):
    model = WaveformModel.from_pretrained(weights_path, dtype=torch.float32)
    model.network.requires_grad_(True).train()
    x = torch.tensor([2., .2, -.1, .3, -.05, .1, -.2], requires_grad=True)
    t = torch.tensor([-900., -500., -200., -160., -80., 0., 40.], requires_grad=True)
    # Exercise the functional attention path used on MPS, on CPU as an arithmetic control.
    import merger_transformer._network as network
    from functools import partial
    original = network.attend_plain
    head_forward = model.network.mode_head.forward
    model.network.mode_head.forward = partial(head_forward, attend=attend_unfused)
    model.network.mode_head.chunk = 256
    network.attend_plain = attend_unfused
    try:
        variables = (x, t, *model.network.parameters())
        ri = model.strain_ri(x, t)
        loss_real = ri.square().sum()
        grads_real = torch.autograd.grad(loss_real, variables)
        loss_complex = model(x, t).abs().square().sum()
        grads_complex = torch.autograd.grad(loss_complex, variables)
        torch.testing.assert_close(loss_real, loss_complex, rtol=2e-6, atol=1e-8)
        a = torch.cat([v.flatten() for v in grads_real])
        b = torch.cat([v.flatten() for v in grads_complex])
        assert torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b) < 3e-6
        with torch.no_grad():
            target = 1.03 * model(x.detach(), t.detach()).numpy()
        ts = t.detach().numpy().astype(float)
        weights = loss_weights(ts, target, np.ones(target.shape, dtype=bool))
        row = Row('test', 'test', 'train', x.detach().numpy(), ts, target, weights)
        torch.testing.assert_close(row_loss_ri(model, row), row_loss(model, row), rtol=1e-5, atol=1e-9)
        before = model.network.conditioner[0].weight.detach().clone()
        opt = torch.optim.AdamW(model.network.parameters(), lr=1e-6, foreach=False, fused=False)
        row_loss_ri(model, row).backward()
        opt.step()
        assert not torch.equal(before, model.network.conditioner[0].weight)
        assert all(torch.isfinite(p).all() for p in model.network.parameters())
    finally:
        network.attend_plain = original
        model.network.mode_head.forward = head_forward


def test_unfused_attention():
    mha = torch.nn.MultiheadAttention(16, 4, batch_first=True, dropout=0.).double().eval()
    q = torch.randn(2, 7, 16, dtype=torch.float64)
    k = torch.randn(2, 3, 16, dtype=torch.float64)
    with torch.no_grad():
        for memory in (q, k):
            torch.testing.assert_close(attend_unfused(mha, q, memory), mha(q, memory, memory, need_weights=True)[0],
                                       rtol=1e-12, atol=1e-14)


def test_mps_precision_guard(weights_path):
    with pytest.raises(ValueError, match='requires float32'):
        WaveformModel.from_pretrained(weights_path, device='mps', dtype=torch.float64)
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match='unavailable'):
            WaveformModel.from_pretrained(weights_path, device='mps')


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires actual Apple Metal hardware')
def test_actual_mps_forward_and_backward(weights_path):
    m = WaveformModel.from_pretrained(weights_path, device='mps', microbatch_size=1)
    cpu = WaveformModel.from_pretrained(weights_path)
    x = torch.tensor([2., .2, -.1, .3, -.05, .1, -.2], device='mps', requires_grad=True)
    t = torch.linspace(-1000, 40, 105, device='mps', requires_grad=True)
    h, p = m.waveforms_ri(x, t)
    assert h.device.type == p.device.type == 'mps'
    hc, pc = cpu.waveforms(x.detach().cpu().double(), t.detach().cpu().double())
    for actual, ref in ((as_complex(h.cpu()), hc), (as_complex(p.cpu()), pc)):
        assert torch.linalg.vector_norm(actual-ref)/torch.linalg.vector_norm(ref) < 5e-4
    assert all(torch.isfinite(v).all() for v in torch.autograd.grad(h.square().sum(), (x,t)))
    out = m(x, t)
    assert out.device.type == 'cpu' and out.dtype == torch.complex64
    with pytest.raises(ValueError, match='CPU/CUDA'):
        m.compile_inference()
