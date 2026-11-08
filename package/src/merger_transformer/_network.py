"""Self-contained numerical core of the selected strain-only v7 model.

All 21 signed ell=2..4 modes are retained. One shared coefficient Transformer
is evaluated on the four exchange/reflection inputs, projected back with the
fixed group action, and multiplied by the recorded symmetric-mass-ratio factor.
There is no phase canonicalization, target input, baseline, or output filtering.
Only the production architecture is supported; its parameter names match the
selected checkpoint's EMA exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ._basis import QuinticBasis


MODES = tuple((ell, m) for ell in range(2, 5) for m in range(-ell, ell + 1))
GROUP = ("I", "E", "P", "EP")
SUBGROUPS = {"both": GROUP}
REFERENCE_CONTRACT = "fixed_physical_reference_n_Lcrossn_L_v1"
VARIANT = "final_v1_nu_factor"
DEFAULT_NU0 = 0.05
MODE_LAYERS = 2
FF_RATIO = 2
_SEED_SALT = 0x6d6f64655f636f75


@dataclass(frozen=True)
class ModelConfig:
    """The complete model_config recorded by the selected product checkpoint."""

    architecture: str = "transformer"
    representation: str = "direct"
    basis_size: int = 384
    time_min: float = -1100.0
    time_max: float = 80.0
    amplitude_scale: float = 0.3
    mlp_width: int = 128
    mlp_depth: int = 4
    transformer_width: int = 192
    transformer_depth: int = 3
    memory_tokens: int = 8
    attention_heads: int = 4
    conditioner_width: int = 256
    conditioner_depth: int = 3
    knot_kind: str = "density"
    density_ratio: float = 4.0
    density_center: float = -200.0
    density_width: float = 40.0
    variant: str = VARIANT
    symmetry: str = "both"
    reference_contract: str = REFERENCE_CONTRACT
    mode_width: int = 32
    nu0: float = DEFAULT_NU0


def coordinates(params: torch.Tensor) -> torch.Tensor:
    """Smooth seven-dimensional chart for positive q, without spin norms/cuts."""
    if params.ndim != 2 or params.shape[1] != 7:
        raise ValueError("params must have shape [batch,7]")
    if not bool(torch.all(torch.isfinite(params))) or bool(torch.any(params[:, 0] <= 0)):
        raise ValueError("parameters must be finite with positive mass ratio")
    return torch.cat(((torch.log(params[:, :1]) / math.log(4.0) - 0.5), params[:, 1:]), dim=1)


class CrossBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.ff_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, query, memory):
        memory = self.memory_norm(memory)
        # Request weights to use explicit softmax/matmul rather than fused SDPA;
        # the latter lacks second input derivatives on the installed CPU build.
        # Memory has only eight tokens, so this does not create a large T x T map.
        query = query + attend_plain(self.attention, self.query_norm(query), memory)
        return query + self.ff(self.ff_norm(query))


class CoefficientSelfAttention(nn.Module):
    """Noncausal, zero-output-initialized attention over coefficient positions."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True,
        )
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(query)
        # Explicit softmax/matmul path: need_weights=True avoids the fused SDPA
        # path whose double backward is unavailable in the installed backend.
        update = attend_plain(self.attention, normalized, normalized)
        return query + update


def _validate_params(params: torch.Tensor) -> None:
    if params.ndim != 2 or params.shape[-1] != 7 or not params.is_floating_point():
        raise ValueError("params must be a real floating [batch,7] tensor")
    if not bool(torch.isfinite(params).all()) or bool((params[:, 0] <= 0).any()):
        raise ValueError("params must be finite with q>0; no q sorting is performed")


def transform_inputs(params: torch.Tensor, element: str) -> torch.Tensor:
    """Return the smooth action on (q, chi1xyz, chi2xyz), including q<1."""
    _validate_params(params)
    if element not in GROUP:
        raise ValueError(f"Unknown group element {element!r}")
    if element == "I":
        return params
    # Exact signs rather than sin(pi)/cos(pi) avoid artificial tiny tilts.
    turn = params.new_tensor((-1., -1., 1.))
    out = params
    if "E" in element:
        out = torch.cat((params[:, :1].reciprocal(),
                         params[:, 4:7] * turn, params[:, 1:4] * turn), dim=-1)
    if "P" in element:
        out = torch.cat((out[:, :1], out[:, 1:4] * turn,
                         out[:, 4:7] * turn), dim=-1)
    return out


def transform_modes(values: torch.Tensor, element: str) -> torch.Tensor:
    """Apply D_g to complex [...,21,time_or_basis] without changing time.

    All group elements are self-inverse and commute. P and EP are real-linear,
    not complex-linear, because they include complex conjugation.
    """
    if element not in GROUP:
        raise ValueError(f"Unknown group element {element!r}")
    if values.ndim < 2 or values.shape[-2] != len(MODES) or not values.is_complex():
        raise ValueError("values must be complex with the complete 21-mode axis at -2")
    out = values
    if "E" in element:
        sign_m = values.real.new_tensor([(-1.) ** m for _, m in MODES])[:, None]
        out = out * sign_m
    if "P" in element:
        reverse = torch.tensor([MODES.index((ell, -m)) for ell, m in MODES],
                               dtype=torch.long, device=values.device)
        sign_ell = values.real.new_tensor([(-1.) ** ell for ell, _ in MODES])[:, None]
        out = out.index_select(-2, reverse).conj() * sign_ell
    return out


def input_orbit(params: torch.Tensor, symmetry: str = "both") -> torch.Tensor:
    """Stack complete group batches for one shared coefficient-network call."""
    try:
        elements = SUBGROUPS[symmetry]
    except KeyError as exc:
        raise ValueError(f"Unknown symmetry {symmetry!r}") from exc
    return torch.cat([transform_inputs(params, g) for g in elements], dim=0)


def project_orbit(orbit_values: torch.Tensor, batch_size: int,
                  symmetry: str = "both") -> torch.Tensor:
    """Reynolds average (1/|G|) sum_g D_g^-1 f(g x), real-linearly."""
    try:
        elements = SUBGROUPS[symmetry]
    except KeyError as exc:
        raise ValueError(f"Unknown symmetry {symmetry!r}") from exc
    if batch_size < 1 or orbit_values.shape[0] != len(elements) * batch_size:
        raise ValueError("Orbit batch size does not match the subgroup")
    chunks = orbit_values.split(batch_size, dim=0)
    terms = [transform_modes(value, g) for g, value in zip(elements, chunks)]
    # Fixed addition pattern, no parameter-dependent branches or mode gates.
    if len(terms) == 4:
        return ((terms[0] + terms[1]) + (terms[2] + terms[3])) * .25
    if len(terms) == 2:
        return (terms[0] + terms[1]) * .5
    return terms[0]


def nu_factor(q: torch.Tensor, nu0: float) -> torch.Tensor:
    """Fixed s(nu), differentiable in q, with no trainable parameters."""
    nu = q / (1 + q) ** 2
    return (1 + (nu0 / nu) ** 4) ** -.25


def attend_unfused(mha, query, memory):
    """Functional real attention, bypassing the inference-only native fast path."""
    q = query.transpose(0, 1)
    k = q if query is memory else memory.transpose(0, 1)
    output, _ = F.multi_head_attention_forward(
        q, k, k, mha.embed_dim, mha.num_heads, mha.in_proj_weight,
        mha.in_proj_bias, mha.bias_k, mha.bias_v, mha.add_zero_attn,
        mha.dropout, mha.out_proj.weight, mha.out_proj.bias,
        training=mha.training, need_weights=True)
    return output.transpose(0, 1)


def attend_plain(mha, query, memory):
    """Explicit softmax/matmul, as the parent's plain path."""
    if query.device.type == "mps":
        return attend_unfused(mha, query, memory)
    return mha(query, memory, memory, need_weights=True)[0]


class ModeLayer(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, FF_RATIO * width), nn.GELU(), nn.Linear(FF_RATIO * width, width))

    def forward(self, tokens, attend):
        normalized = self.attention_norm(tokens)
        tokens = tokens + attend(self.attention, normalized, normalized)
        return tokens + self.ff(self.ff_norm(tokens))


class ModeCouplingHead(nn.Module):
    """[..., z_width] -> [..., 21, 2]; each sequence is one time token of one orbit row."""

    def __init__(self, z_width: int, width: int, heads: int, chunk: int):
        super().__init__()
        self.chunk = chunk
        self.project = nn.Linear(z_width, width)
        # Comparable to the projected z (std about 0.58), so the mode tokens start distinct.
        self.embedding = nn.Parameter(torch.empty(len(MODES), width))
        nn.init.normal_(self.embedding, std=0.5)
        self.layers = nn.ModuleList(ModeLayer(width, heads) for _ in range(MODE_LAYERS))
        self.final_norm = nn.LayerNorm(width)
        # Per-mode readout. Zero weights and bias: output exactly zero at initialization.
        self.readout_weight = nn.Parameter(torch.zeros(len(MODES), width, 2))
        self.readout_bias = nn.Parameter(torch.zeros(len(MODES), 2))

    def sequences(self, z, attend):
        tokens = self.project(z)[:, None] + self.embedding
        for layer in self.layers:
            tokens = layer(tokens, attend)
        return torch.einsum("smd,mdo->smo", self.final_norm(tokens), self.readout_weight) + self.readout_bias

    def forward(self, z, attend=attend_plain):
        flat = z.reshape(-1, z.shape[-1])
        if self.chunk <= 0 or len(flat) <= self.chunk:
            out = self.sequences(flat, attend)
        else:
            out = torch.cat([self.sequences(piece, attend) for piece in flat.split(self.chunk)])
        return out.reshape(*z.shape[:-1], len(MODES), 2)


class Generator(nn.Module):
    """Float64 EMA-compatible coefficient network and differentiable basis.

    Construction preserves the original parameter names and order. The caller
    supplies weights and device/dtype; the production wrapper uses float64.
    mode_head.chunk defaults to zero so higher derivatives retain an ordinary
    autograd graph. Positive chunk sizes use the same ordinary operations.
    """

    def __init__(self, config: ModelConfig | dict | None = None):
        super().__init__()
        config = ModelConfig() if config is None else config
        if isinstance(config, dict):
            config = ModelConfig(**config)
        if not isinstance(config, ModelConfig) or config != ModelConfig():
            raise ValueError("Only the recorded selected v7 configuration is supported")
        self.config = config
        self.basis = QuinticBasis()
        layers, incoming = [], 7
        for _ in range(config.conditioner_depth):
            layers.extend((nn.Linear(incoming, config.conditioner_width), nn.SiLU()))
            incoming = config.conditioner_width
        layers.append(nn.Linear(incoming, config.memory_tokens * config.transformer_width))
        self.conditioner = nn.Sequential(*layers)
        self.query_embedding = nn.Parameter(torch.empty(config.basis_size, config.transformer_width))
        nn.init.normal_(self.query_embedding, std=0.02)
        self.blocks = nn.ModuleList(CrossBlock(config.transformer_width, config.attention_heads)
                                    for _ in range(config.transformer_depth))
        self.final_norm = nn.LayerNorm(config.transformer_width)
        self.output = nn.Linear(config.transformer_width, 42)
        nn.init.normal_(self.output.weight, std=0.001)
        nn.init.zeros_(self.output.bias)
        with torch.random.fork_rng(devices=[]):
            self.coefficient_attention = nn.ModuleList(
                CoefficientSelfAttention(config.transformer_width, config.attention_heads)
                for _ in range(config.transformer_depth)
            )
        with torch.random.fork_rng(devices=[]):
            salted = int(torch.randint(2**62, (1,))) ^ _SEED_SALT
            torch.default_generator.manual_seed(salted)
            self.mode_head = ModeCouplingHead(config.transformer_width, config.mode_width,
                                              config.attention_heads, chunk=0)

    def config_dict(self):
        return asdict(self.config)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def _raw_output(self, params: torch.Tensor) -> torch.Tensor:
        c = self.config
        memory = self.conditioner(coordinates(params)).reshape(len(params), c.memory_tokens, c.transformer_width)
        query = self.query_embedding[None].expand(len(params), -1, -1)
        for block, coefficient_attention in zip(self.blocks, self.coefficient_attention):
            query = coefficient_attention(block(query, memory))
        z = self.final_norm(query)
        out = self.output(z).reshape(len(params), c.basis_size, 21, 2) + self.mode_head(z)
        return out

    def raw_coefficients(self, params: torch.Tensor) -> torch.Tensor:
        c = self.config
        out = self._raw_output(params)
        return torch.complex(out[..., 0], out[..., 1]).transpose(1, 2) * c.amplitude_scale

    def coefficients_ri(self, params: torch.Tensor) -> torch.Tensor:
        """Real [B,21,384,2] coefficients; no complex operations on device."""
        if params.shape[0] == 0:
            raise ValueError("Empty batches are not supported")
        orbit = input_orbit(params, self.config.symmetry)
        values = self._raw_output(orbit).transpose(1, 2) * self.config.amplitude_scale
        chunks = values.split(len(params), dim=0)
        reverse = torch.tensor([MODES.index((ell, -m)) for ell, m in MODES],
                               device=params.device, dtype=torch.long)
        sm = params.new_tensor([(-1.)**m for _, m in MODES])[:, None, None]
        sl = params.new_tensor([(-1.)**ell for ell, _ in MODES])[:, None, None]
        conjugation = params.new_tensor([1., -1.])
        aa, bb, cc, dd = chunks
        bb = bb * sm
        cc = cc.index_select(-3, reverse) * conjugation * sl
        dd = (dd * sm).index_select(-3, reverse) * conjugation * sl
        projected = ((aa + bb) + (cc + dd)) * .25
        return projected * nu_factor(params[:, 0], self.config.nu0)[:, None, None, None]

    def coefficients(self, params: torch.Tensor) -> torch.Tensor:
        if params.shape[0] == 0:
            raise ValueError("Empty batches are not supported")
        orbit = input_orbit(params, self.config.symmetry)
        values = self.raw_coefficients(orbit)
        projected = project_orbit(values, len(params), self.config.symmetry)
        return projected * nu_factor(params[:, 0], self.config.nu0)[:, None, None]

    def forward_precomputed(self, params, basis_h, basis_Psi4):
        coefficients = self.coefficients(params)
        h = coefficients @ basis_h.to(coefficients.dtype)
        psi = coefficients @ basis_Psi4.to(coefficients.dtype)
        return {"h": h, "Psi4": psi, "coefficients": coefficients}

    def forward(self, params, t_h, t_psi=None):
        if t_psi is None:
            t_psi = t_h
        bh, _, _ = self.basis(t_h)
        _, _, bp = self.basis(t_psi)
        return self.forward_precomputed(params, bh, -bp)


__all__ = ["Generator", "ModelConfig", "MODES", "REFERENCE_CONTRACT"]
