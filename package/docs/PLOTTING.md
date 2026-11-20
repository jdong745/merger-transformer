# Waveform plotting and array conventions

From `package/`, run:

```bash
python -m merger_transformer.plotting ../weights/weights.pt --config examples/plot_config.json --output output/example
```

Defaults use `time_min_M = -1500.0`, `time_max_M = 100.0`, and `dt_M = 0.5` (within native support `[-1620, 160]M`), generating:
- `<output>_polarizations.png/.pdf`: observer-projected `h_+`, `h_x` and analytic curvature `Psi_4 = -d^2 h / dtau^2`.
- `<output>_modes.png/.pdf`: real and imaginary parts of all 21 signed strain modes.
- `<output>_psi4_modes.png/.pdf`: all 21 signed analytic curvature modes.
- `<output>.npz` and `<output>_settings.json`: full complex arrays and metadata.

Add `"mass_msun"` and `"distance_mpc"` to the JSON to also export and plot in SI physical units (`t_s`, `strain`, `s^-2`) at detector-frame total mass `M` and luminosity distance `D`, holding `M f_ref = 0.01182117827433904` fixed.
