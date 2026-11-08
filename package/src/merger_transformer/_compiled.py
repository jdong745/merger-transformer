"""Guard-free arithmetic adapter, matched independently to official eager inference."""
import math
import torch
from ._network import MODES

class _Arithmetic(torch.nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('turn', torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
        self.register_buffer('sm', torch.tensor([(-1.0) ** m for (l, m) in MODES], dtype=torch.float64)[:, None])
        self.register_buffer('sl', torch.tensor([(-1.0) ** l for (l, m) in MODES], dtype=torch.float64)[:, None])
        self.register_buffer('reverse', torch.tensor([MODES.index((l, -m)) for (l, m) in MODES], dtype=torch.long))

    def coefficients(self, x):
        t = self.turn
        e = torch.cat((x[:, :1].reciprocal(), x[:, 4:7] * t, x[:, 1:4] * t), dim=-1)
        p = torch.cat((x[:, :1], x[:, 1:4] * t, x[:, 4:7] * t), dim=-1)
        ep = torch.cat((e[:, :1], e[:, 1:4] * t, e[:, 4:7] * t), dim=-1)
        x4 = torch.cat((x, e, p, ep), dim=0)
        n = self.net
        c = n.config
        coord = torch.cat((torch.log(x4[:, :1]) / math.log(4.0) - 0.5, x4[:, 1:]), dim=1)
        mem = n.conditioner(coord).reshape(len(x4), c.memory_tokens, c.transformer_width)
        q = n.query_embedding[None].expand(len(x4), -1, -1)
        for (block, attn) in zip(n.blocks, n.coefficient_attention):
            q = attn(block(q, mem))
        z = n.final_norm(q)
        out = n.output(z).reshape(len(x4), c.basis_size, 21, 2) + n.mode_head(z)
        raw = torch.complex(out[..., 0], out[..., 1]).transpose(1, 2) * c.amplitude_scale
        (aa, bb, cc, dd) = raw.split(len(x), dim=0)
        bb = bb * self.sm
        cc = cc.index_select(-2, self.reverse).conj() * self.sl
        dd = (dd * self.sm).index_select(-2, self.reverse).conj() * self.sl
        coeff = (aa + bb + (cc + dd)) * 0.25
        nu = x[:, 0] / (1 + x[:, 0]) ** 2
        factor = (1 + (c.nu0 / nu) ** 4) ** (-0.25)
        return coeff * factor[:, None, None]

    def basis(self, times):
        knots = self.net.basis.knots.to(device=times.device, dtype=times.dtype)
        x = times[None, :]
        rightmost = (knots[1:] == self.net.basis.high) & (knots[:-1] < self.net.basis.high)
        level = ((x >= knots[:-1, None]) & (x < knots[1:, None]) | (x == self.net.basis.high) & rightmost[:, None]).to(times.dtype)
        d1 = torch.zeros_like(level)
        d2 = torch.zeros_like(level)
        for degree in range(1, 6):
            size = len(knots) - degree - 1
            dl = knots[degree:degree + size] - knots[:size]
            dr = knots[degree + 1:degree + size + 1] - knots[1:size + 1]
            il = torch.where(dl != 0, 1 / torch.where(dl != 0, dl, torch.ones_like(dl)), 0)[:, None]
            ir = torch.where(dr != 0, 1 / torch.where(dr != 0, dr, torch.ones_like(dr)), 0)[:, None]
            nl = (x - knots[:size, None]) * il * level[:size] + (knots[degree + 1:degree + size + 1, None] - x) * ir * level[1:size + 1]
            nd1 = degree * (il * level[:size] - ir * level[1:size + 1])
            nd2 = degree * (il * d1[:size] - ir * d1[1:size + 1])
            (level, d1, d2) = (nl, nd1, nd2)
        return (level, d1, d2)

    def forward(self, x, times):
        (bh, _, _) = self.basis(times)
        return self.coefficients(x) @ bh.to(torch.complex128 if times.dtype == torch.float64 else torch.complex64)


class _Coefficients(torch.nn.Module):
    def __init__(self, arithmetic):
        super().__init__()
        self.arithmetic = arithmetic
    def forward(self, params):
        return self.arithmetic.coefficients(params)


class _Basis(torch.nn.Module):
    def __init__(self, arithmetic, derivative):
        super().__init__()
        self.arithmetic = arithmetic
        self.derivative = derivative
    def forward(self, times):
        value = self.arithmetic.basis(times)[self.derivative]
        return value.to(torch.complex128 if times.dtype == torch.float64 else torch.complex64)


class CompiledInference:
    """Optional frozen Inductor forward-only snapshot of an eager model.

    This is lazy: the first call for every new shape/stride/device/precision may
    compile C++/Triton kernels. Compile and warm up before measuring throughput.
    CPU strain uses one full graph. GPU and derivative paths compile the
    coefficient and basis components; validation, microbatch orchestration and
    complex matrix multiplication remain ordinary timed PyTorch operations.
    No basis is cached and no physical input is fixed. Compilation errors raise;
    there is no silent fallback. Use the original model for automatic derivatives.
    Tested with PyTorch 2.12.1; the compiler is an optional versioned dependency.
    """
    def __init__(self, model):
        import copy
        import threading
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable; use the eager model")
        major_minor = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
        if major_minor != (2, 12):
            raise RuntimeError("Frozen compilation is supported here with PyTorch 2.12.x; eager remains available")
        if model.device.type not in ("cpu", "cuda"):
            raise ValueError("Frozen compilation supports CPU and CUDA")
        if model.dtype != torch.float64:
            raise ValueError("The audited frozen path requires float64; use eager for float32")
        self._model = copy.deepcopy(model).eval().requires_grad_(False)
        self._arithmetic = _Arithmetic(self._model.network).to(device=model.device, dtype=model.dtype).eval().requires_grad_(False)
        self._lock = threading.RLock()
        self._full = None
        self._coeff = None
        self._bases = {}
        self.metadata = dict(model.metadata)
        self.metadata.update(compilation="torch.compile/Inductor freezing", forward_only=True,
                             full_cpu_strain_graph=True, runtime_inputs=True, cached_basis=False)

    @property
    def mode_order(self): return self._model.mode_order
    @property
    def device(self): return self._model.device
    @property
    def dtype(self): return self._model.dtype
    @property
    def time_domain(self): return self._model.time_domain

    @staticmethod
    def _compile(module):
        return torch.compile(module, backend="inductor", fullgraph=True, dynamic=False)

    def _component(self, params, times, derivative):
        if self._coeff is None:
            self._coeff = self._compile(_Coefficients(self._arithmetic))
        if derivative not in self._bases:
            self._bases[derivative] = self._compile(_Basis(self._arithmetic, derivative))
        basis = self._bases[derivative](times)
        parts = [self._coeff(x) @ basis for x in params.split(self._model.microbatch_size)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def evaluate(self, params, times, *, derivative=0):
        """Generate h, dh/dtau or d2h/dtau2; intentionally no autograd output."""
        import torch._inductor.config as config
        if derivative not in (0, 1, 2):
            raise ValueError("derivative must be 0, 1 or 2")
        # Reject AD requests explicitly, rather than silently detaching inputs.
        if any(isinstance(v, torch.Tensor) and v.requires_grad for v in (params, times)):
            raise ValueError("CompiledInference is forward-only; use the eager model for input/time gradients")
        with self._lock, torch.no_grad(), config.patch(freezing=True, freezing_discard_parameters=False):
            x, single = self._model._prepare_parameters(params)
            t = self._model._prepare_times(times)
            if self.device.type == "cpu" and derivative == 0:
                if self._full is None:
                    self._full = self._compile(self._arithmetic)
                parts = [self._full(piece, t) for piece in x.split(self._model.microbatch_size)]
                result = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            else:
                result = self._component(x, t, derivative)
            return result[0] if single else result

    def __call__(self, params, times):
        return self.evaluate(params, times)

    def psi4(self, params, times):
        return -self.evaluate(params, times, derivative=2)

    def waveforms(self, params, times):
        """Return h and Psi4 (separate compiled evaluations)."""
        return self(params, times), self.psi4(params, times)
