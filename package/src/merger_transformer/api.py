"""Public tensor API. Input and time derivatives remain attached to autograd."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ._network import Generator, ModelConfig, MODES

WEIGHT_FORMAT = "merger-transformer-inference-v1"
ADAPTED_WEIGHT_FORMAT = "merger-transformer-private-finetune-v1"


class WaveformModel(nn.Module):
    """A frozen coefficient network with continuously evaluated quintic splines.

    Inputs are [q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z], either one
    seven-vector or a batch. Times are a common one-dimensional array in M.
    Outputs have shape [21, T] or [batch, 21, T], following MODES.
    """

    def __init__(self, network: Generator, *, microbatch_size: int = 4,
                 metadata: dict[str, Any] | None = None):
        super().__init__()
        if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int) or microbatch_size < 1:
            raise ValueError("microbatch_size must be a positive integer")
        self.network = network.eval().requires_grad_(False)
        self.microbatch_size = microbatch_size
        self.metadata = dict(metadata or {})
        self.eval()

    @classmethod
    def from_pretrained(cls, weights_path: str | Path, *, device: str | torch.device = "cpu",
                        dtype: torch.dtype = torch.float64, microbatch_size: int = 4) -> "WaveformModel":
        """Load an inference-only weights.pt file without importing training code.

        float64 reproduces the paper's EMA inference. float32 is optional and
        has lower numerical precision. No file downloads occur implicitly.
        """
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use torch.float64 (reference precision) or torch.float32")
        payload = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") not in (WEIGHT_FORMAT, ADAPTED_WEIGHT_FORMAT):
            raise ValueError("Expected an exported merger-transformer inference weights file")
        if set(payload) != {"format", "model_config", "state_dict", "metadata"}:
            raise ValueError("Unexpected weights fields; use the inference-only export")
        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Expected weights metadata")
        base = metadata if payload["format"] == WEIGHT_FORMAT else metadata.get("base_product", {})
        if (not isinstance(base, dict) or base.get("model") != "v7_width192_constant_low_lr"
                or base.get("selected_step") != 48000
                or base.get("continuation_arm") != "constant"
                or base.get("continuation_start_step") != 40000
                or base.get("continuation_learning_rate") != 1e-5):
            raise ValueError("This release requires v7 width192 constant low-LR STEP048000 EMA weights")
        if payload["format"] == ADAPTED_WEIGHT_FORMAT and (
            metadata.get("model") != "private_nr_finetune"
            or not isinstance(metadata.get("adaptation"), dict)
        ):
            raise ValueError("Missing private fine-tuning provenance")
        if payload["metadata"].get("weight_kind") != "EMA":
            raise ValueError("The released model requires EMA weights")
        # Construction must not advance the caller's random stream.
        with torch.random.fork_rng(devices=[]):
            network = Generator(ModelConfig(**payload["model_config"])).double()
        state = payload["state_dict"]
        if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all()) for v in state.values()):
            raise ValueError("All model state entries must be finite tensors")
        if "basis.knots" not in state or not torch.equal(state["basis.knots"], network.basis.knots):
            raise ValueError("Spline knots do not match the released model configuration")
        network.load_state_dict(state, strict=True)
        network = network.to(device=device, dtype=dtype)
        return cls(network, microbatch_size=microbatch_size, metadata=payload["metadata"])

    @property
    def device(self) -> torch.device:
        return self.network.basis.knots.device

    @property
    def dtype(self) -> torch.dtype:
        return self.network.basis.knots.dtype

    @property
    def mode_order(self) -> tuple[tuple[int, int], ...]:
        return MODES

    @property
    def time_domain(self) -> tuple[float, float]:
        return (self.network.config.time_min, self.network.config.time_max)

    def _real_tensor(self, value: Any, name: str) -> torch.Tensor:
        probe = torch.as_tensor(value)
        if probe.is_complex():
            raise ValueError(f"{name} must be real")
        # Construct from the original value: casting a default-float32 probe
        # would discard digits in Python float lists before float64 inference.
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def _prepare_parameters(self, params: Any) -> tuple[torch.Tensor, bool]:
        x = self._real_tensor(params, "params")
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != 7 or x.shape[0] == 0:
            raise ValueError("params must have shape [7] or [nonempty batch, 7]")
        if not bool(torch.isfinite(x).all()) or bool((x[:, 0] <= 0).any()):
            raise ValueError("Parameters must be finite with q > 0")
        tolerance = 16 * torch.finfo(self.dtype).eps
        if bool((torch.linalg.vector_norm(x[:, 1:4], dim=-1) > 1 + tolerance).any()) or bool(
            (torch.linalg.vector_norm(x[:, 4:7], dim=-1) > 1 + tolerance).any()
        ):
            raise ValueError("Each dimensionless black-hole spin must have magnitude <= 1")
        return x, single

    def _prepare_times(self, times: Any) -> torch.Tensor:
        t = self._real_tensor(times, "times")
        if t.ndim != 1 or t.numel() == 0 or not bool(torch.isfinite(t).all()):
            raise ValueError("times must be a finite, nonempty one-dimensional array")
        lo, hi = self.time_domain
        if bool(((t < lo) | (t > hi)).any()):
            raise ValueError(f"Times must lie in [{lo}, {hi}] M; no time extrapolation or clipping is applied")
        return t

    def coefficients(self, params: Any) -> torch.Tensor:
        """Return complex [21,384] or [batch,21,384] spline coefficients."""
        x, single = self._prepare_parameters(params)
        chunks = [self.network.coefficients(piece) for piece in x.split(self.microbatch_size)]
        c = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        return c[0] if single else c

    def evaluate_coefficients(self, coefficients: torch.Tensor, times: Any, *, derivative: int = 0) -> torch.Tensor:
        """Evaluate h or its first/second time derivative from cached coefficients.

        derivative=2 means d2h/dtau2. Psi4 has the opposite sign.
        The coefficient tensor's autograd graph is retained.
        """
        if derivative not in (0, 1, 2):
            raise ValueError("derivative must be 0, 1 or 2")
        t = self._prepare_times(times)
        c = torch.as_tensor(coefficients, device=self.device)
        if c.ndim not in (2, 3) or c.shape[-2:] != (21, 384) or not c.is_complex():
            raise ValueError("coefficients must be complex with shape [21,384] or [batch,21,384]")
        if c.ndim == 3 and c.shape[0] == 0:
            raise ValueError("Empty coefficient batches are not supported")
        complex_dtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
        c = c.to(dtype=complex_dtype)
        basis = self.network.basis(t)[derivative].to(dtype=complex_dtype)
        return c @ basis

    def forward(self, params: Any, times: Any) -> torch.Tensor:
        """Generate dimensionless strain modes h_lm, preserving input gradients."""
        t = self._prepare_times(times)
        return self.evaluate_coefficients(self.coefficients(params), t)

    def psi4(self, params: Any, times: Any) -> torch.Tensor:
        """Generate dimensionless Psi4_lm = -d2h_lm/dtau2 analytically."""
        t = self._prepare_times(times)
        return -self.evaluate_coefficients(self.coefficients(params), t, derivative=2)

    def waveforms(self, params: Any, times: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (strain, Psi4), evaluating the coefficient network only once."""
        t = self._prepare_times(times)
        c = self.coefficients(params)
        b, _, b2 = self.network.basis(t)
        return c @ b.to(c.dtype), -(c @ b2.to(c.dtype))

    def mode_dict(self, params: Any, times: Any, *, field: str = "strain") -> dict[tuple[int, int], torch.Tensor]:
        """Return {(ell,m): mode} with the same single/batch convention."""
        if field not in ("strain", "psi4"):
            raise ValueError("field must be 'strain' or 'psi4'")
        values = self(params, times) if field == "strain" else self.psi4(params, times)
        return {mode: values[..., index, :] for index, mode in enumerate(MODES)}

    def compile_inference(self):
        """Create an independent frozen, forward-only Inductor snapshot.

        Compilation is lazy and shape-specialized. No eager behavior is changed.
        Use this model, not the returned object, for input/time autodifferentiation.
        """
        from ._compiled import CompiledInference
        return CompiledInference(self)
