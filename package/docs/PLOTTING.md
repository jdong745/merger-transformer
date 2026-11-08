# Waveform plotting and array conventions

From the installed code directory, run:

```bash
python -m merger_transformer.plotting ../weights/weights.pt --config examples/plot_config.json --output output/example
```

The input is one seven-vector in the order `(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)`. Spin components are given at `M f_ref=0.01182117827433904` in `(n,Lhat × n,Lhat)`, with `n` pointing from body 2 to body 1. `inclination_rad` is measured from this z axis; `azimuth_rad` from its x axis. Neither angle is an alignment fit. The model's residual reference-time/azimuth error is not removed by the plotting command.

The unscaled exported arrays are:

| NPZ field | Meaning |
| --- | --- |
| `time_M` | Native dimensionless time tau; endpoints as requested |
| `parameters` | Exact supplied seven-vector |
| `modes` | 21 `(ell,m)` pairs, ascending ell then m |
| `strain_modes` | Complex H_lm of shape `[21,T]` |
| `psi4_modes` | Complex P_lm = -d²H_lm/dtau², shape `[21,T]` |
| `projected_strain` | Sum H_lm times spin-weight -2 Y_lm(theta,phi) |
| `projected_psi4` | Corresponding sum of P_lm |

`h_plus=Re(projected_strain)` and `h_cross=-Im(projected_strain)`. Signed negative-m and odd-m modes are used as generated; no reflection completion or parity approximation is applied. The finite-sum harmonic implementation follows the LAL spin-weighted spherical-harmonic convention. It needs only PyTorch and Python's standard library. The optional plotting command needs NumPy and Matplotlib.

To make physical-unit plots, add both fields to the JSON:

```json
"mass_msun": 120.0,
"distance_mpc": 1000.0
```

Here M is total **detector-frame** mass and D is luminosity distance. Define `M_seconds=M*4.9254909476412675e-6`, `M_length=c*M_seconds`. Then `t_seconds=tau*M_seconds`, physical strain is `H*M_length/D`, and physical curvature in s^-2 is `P*(M_length/D)/M_seconds²`. No extra factors of q, nu, two, mass, or distance are needed. The observer projection commutes with these scalar conversions. Physical arrays are added to the NPZ; the dimensionless ones are retained.

Changing the output mass does not evolve the input spins or reinterpret their dimensionless reference frequency. For example, 60 solar masses corresponds to reference frequency 40 Hz with the same `M f_ref`. The clock is the learned curvature-power time moment, not an enforced strain peak. The finite model window is not a complete 20-Hz inspiral at arbitrary mass.

There is no low-pass filter or numerical time differencing in these plots. The curvature is the exact spline second derivative with the stated minus sign. Smoothness of that representation does not certify physical curvature accuracy. Generated figures illustrate the model only; NR comparisons and their alignments require separately licensed reference data and evaluation procedures.

The native `[-1100,80]M` support includes guard intervals outside the usual reported `[-1000,40]M` strain window (curvature is assessed on `[-160,40]M`). Native support is not an accuracy guarantee at its endpoints: the full-window example can show endpoint oscillations and large analytic-curvature excursions. The plotting command retains these features without filtering or silently cropping them.
