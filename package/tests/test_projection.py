"""Observer projection tests requiring no external harmonic package."""
import math
import pytest
import torch
from merger_transformer import MODES
from merger_transformer.projection import harmonics, project_modes, physical_scales


def test_quadrupole_harmonics_and_poles():
    for theta in [0., .7, math.pi/2, math.pi]:
        phi = .31
        values = harmonics(theta, phi)
        expected = math.sqrt(5/(64*math.pi))*(1+math.cos(theta))**2*complex(math.cos(2*phi),math.sin(2*phi))
        torch.testing.assert_close(values[MODES.index((2,2))], torch.tensor(expected,dtype=torch.complex128), rtol=2e-14,atol=2e-15)


def test_projection_shape_and_parameter_gradients():
    modes = torch.arange(42,dtype=torch.float64).reshape(2,21,1).to(torch.complex128)
    theta = torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    phi = torch.tensor(.31,dtype=torch.float64,requires_grad=True)
    projected = project_modes(modes,theta,phi)
    assert projected.shape == (2,1)
    expected = (modes*harmonics(theta,phi)[:,None]).sum(-2)
    torch.testing.assert_close(projected,expected,rtol=0,atol=0)
    grads = torch.autograd.grad(projected.real.sum(),(theta,phi))
    assert all(torch.isfinite(x) for x in grads)


def test_physical_scaling_and_validation():
    t,h,p = physical_scales(120.,500.)
    tt,hh,pp = physical_scales(240.,500.)
    assert tt == 2*t and hh == 2*h and pp == p/2
    assert p == h/t**2
    with pytest.raises(ValueError): physical_scales(0.,500.)
    with pytest.raises(ValueError): harmonics(-.1,0.)
