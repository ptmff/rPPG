"""Pure-PyTorch drop-in for mamba_ssm.modules.mamba_simple.Mamba (mamba-ssm 2.2.x).

mamba-ssm needs CUDA kernels, so it does not install on macOS. This module keeps the
exact parameter names and shapes (in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj),
so checkpoints trained with mamba-ssm load unchanged, and reproduces the reference
(non-fused) forward pass: mamba_ssm.ops.selective_scan_interface.selective_scan_ref.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def selective_scan(u, delta, A, B, C, D, z, delta_bias):
    """u, delta, z: (b, d, l); A: (d, n); B, C: (b, n, l); D, delta_bias: (d,).
    Same maths as mamba_ssm's selective_scan_ref with delta_softplus=True."""
    delta = F.softplus(delta + delta_bias[None, :, None])
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    x = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    ys = []
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
    y = torch.stack(ys, dim=2)
    y = y + u * D[None, :, None]
    return y * F.silu(z)


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True,
                 layer_idx=None, device=None, dtype=None):
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **kw)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=conv_bias, **kw)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False, **kw)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **kw)

        # Initialisation as in mamba-ssm (only matters for training from scratch)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner, **kw) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        self.A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1).contiguous()))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **kw)

    def forward(self, hidden_states):
        """hidden_states: (B, L, D) -> (B, L, D)"""
        b, l, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)  # (b, 2*d_inner, l)
        x, z = xz.chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :l])
        x_dbl = self.x_proj(x.transpose(1, 2).reshape(b * l, -1))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (self.dt_proj.weight @ dt.t()).reshape(self.d_inner, b, l).permute(1, 0, 2)
        B = B.reshape(b, l, self.d_state).transpose(1, 2)
        C = C.reshape(b, l, self.d_state).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        y = selective_scan(x, dt, A, B, C, self.D.float(), z, self.dt_proj.bias.float())
        return self.out_proj(y.transpose(1, 2))
