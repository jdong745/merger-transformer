"""Observer projection of signed modes. No NR or external harmonic library needed."""
from __future__ import annotations
import math
import torch
from ._network import MODES

# Conventional solar-mass time and parsec in SI; distance is luminosity distance.
MTSUN_SI = 4.9254909476412675e-6
C_SI = 299792458.0
MPC_SI = 3.085677581491367e22


def harmonics(inclination, azimuth=0., *, device=None, dtype=torch.float64):
    """Return all21 spin -2 Y_lm(theta,phi), in MODES order.

    Angles are scalar radians, theta in [0,pi]. The phase convention agrees
    with LAL's SpinWeightedSphericalHarmonic(theta,phi,-2,ell,m).
    Tensor angles retain their autograd graphs. No source-frame fitting occurs.
    """
    theta = torch.as_tensor(inclination, device=device, dtype=dtype)
    phi = torch.as_tensor(azimuth, device=device, dtype=dtype)
    if theta.ndim or phi.ndim or not bool(torch.isfinite(theta)) or not bool(torch.isfinite(phi)):
        raise ValueError('Observer angles must be finite scalars')
    if bool((theta < 0) | (theta > math.pi)):
        raise ValueError('Inclination must lie in [0, pi] radians')
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    values = []
    for ell, m in MODES:
        p = 2
        prefactor = math.sqrt(math.factorial(ell+m)*math.factorial(ell-m)*
                              math.factorial(ell+p)*math.factorial(ell-p))
        d = torch.zeros_like(theta)
        for k in range(max(0, m-p), min(ell+m, ell-p)+1):
            a = (-1)**k * prefactor / (math.factorial(ell+m-k)*math.factorial(k)*
                 math.factorial(p-m+k)*math.factorial(ell-p-k))
            d = d + a*c**(2*ell+m-p-2*k)*s**(p-m+2*k)
        values.append(math.sqrt((2*ell+1)/(4*math.pi))*d*torch.exp(1j*m*phi))
    return torch.stack(values)


def project_modes(modes: torch.Tensor, inclination, azimuth=0.):
    """Sum (...,21,T) modes to h_plus - i*h_cross (or projected Psi4)."""
    if modes.ndim < 2 or modes.shape[-2] != 21 or not modes.is_complex():
        raise ValueError('modes must be a complex tensor with shape (...,21,T)')
    y = harmonics(inclination, azimuth, device=modes.device, dtype=modes.real.dtype)
    return (modes * y[:, None]).sum(dim=-2)


def physical_scales(mass_msun: float, distance_mpc: float):
    """Return (M_seconds, strain_scale, curvature_scale).

    Input M is detector-frame total mass; H=(D/M_length)h, P=-H''(tau).
    This rescales outputs only; it does NOT transport spins or change Mf_ref.
    """
    if not math.isfinite(mass_msun) or mass_msun <= 0 or not math.isfinite(distance_mpc) or distance_mpc <= 0:
        raise ValueError('Mass and luminosity distance must be finite and positive')
    mt = mass_msun*MTSUN_SI
    strain = mt*C_SI/(distance_mpc*MPC_SI)
    return mt, strain, strain/(mt*mt)
