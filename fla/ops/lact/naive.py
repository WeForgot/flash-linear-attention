
"""
Pure-PyTorch reference for the LaCT (Large-Chunk Test-Time Training) SwiGLU operator.

This is the ground truth `chunk_lact_swiglu` and `fused_chunk_lact_swiglu` are tested against, so it
is deliberately written for legibility over speed: no `torch.compile`, no Triton, no fused epilogues.

Unlike the upstream LaCT reference it is **dtype-transparent** — it never forces bfloat16, so it can
serve as an fp32 reference. Every matmul casts its operands to a single compute dtype (taken from the
activations), which is what an ambient `torch.autocast` would have done implicitly upstream.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ['naive_lact_swiglu']

# Newton-Schulz quintic coefficients, from the Muon reference implementation.
_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def silu_backprop(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""
    Vector-Jacobian product of SiLU: `dx = dy * sigma * (1 + x * (1 - sigma))`.

    Args:
        dy (torch.Tensor):
            Gradient of the outer loss w.r.t. the activation output, of shape `[B, D, L]`.
        x (torch.Tensor):
            Pre-activation input, of shape `[B, D, L]`.

    Returns:
        dx (torch.Tensor):
            Gradient w.r.t. `x`, of shape `[B, D, L]`.
    """
    sigma = torch.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    r"""
    Row-wise L2 normalization over the last dim, with eps added *outside* the square root.

    Note this differs from `fla.modules.l2norm`, which folds eps under the root. The placement is
    load-bearing for LaCT's numerics, so it is reproduced exactly.

    Args:
        x (torch.Tensor):
            Input of shape `[..., D]`.

    Returns:
        y (torch.Tensor):
            Normalized tensor of the same shape and dtype as `x`.
    """
    dtype = x.dtype
    # `norm` upcasts to fp32 internally
    return (x / (x.norm(dim=-1, keepdim=True) + 1e-5)).to(dtype)


def zeropower_via_newtonschulz5(G: torch.Tensor) -> torch.Tensor:
    r"""
    Batched Newton-Schulz orthogonalization (the Muon "zeroth power" of `G`).

    Differs from the single-matrix reference in operating on a batch: `G` is `[B, D, D']` rather than
    `[D, D']`. Unlike upstream LaCT this preserves the input dtype instead of forcing bfloat16.

    Args:
        G (torch.Tensor):
            Gradient batch of shape `[B, D, D']`.

    Returns:
        X (torch.Tensor):
            Orthogonalized batch of shape `[B, D, D']`.

    FLOPs:
        When `D == D'`, `30 * B * D ** 3`.
    """
    assert G.ndim == 3, f"expected a 3D batch of matrices, got shape {tuple(G.shape)}"
    X = G
    transposed = G.size(1) > G.size(2)
    if transposed:
        X = X.transpose(1, 2)
    # ensure the spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for a, b, c in _NS_COEFFS:
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.transpose(1, 2)
    return X


def _bmm(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Batched matmul with both operands cast to a common dtype, as autocast would do."""
    return torch.bmm(a.to(dtype), b.to(dtype))


def _apply_muon(dw0, dw1, dw2, muon_group: str):
    """Orthogonalize the fast-weight gradients, either per matrix or with w0/w2 stacked."""
    dw1 = zeropower_via_newtonschulz5(dw1)
    if muon_group == 'per_matrix':
        return zeropower_via_newtonschulz5(dw0), dw1, zeropower_via_newtonschulz5(dw2)
    if muon_group == 'w0w2_joint':
        # what the fused Triton path does natively: one Newton-Schulz on the stacked [2 * d_h, d_in]
        # block. This is NOT equivalent to orthogonalizing each half.
        d_h = dw0.shape[1]
        dw0_dw2 = zeropower_via_newtonschulz5(torch.cat([dw0, dw2], dim=1))
        return dw0_dw2[:, :d_h], dw1, dw0_dw2[:, d_h:]
    raise ValueError(f"muon_group must be 'per_matrix' or 'w0w2_joint', got {muon_group!r}")


def naive_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: torch.Tensor | None = None,
    prenorm: bool = False,
    muon_group: str = 'per_matrix',
) -> torch.Tensor:
    r"""
    Block-causal LaCT with a SwiGLU fast-weight function, `f(x) = w1 @ (silu(w0 @ x) * (w2 @ x))`.

    Each chunk is **applied first** (reading out with the previous chunk's fast weights) and
    **updated second**, which is what makes the operator strictly causal at chunk granularity —
    "apply then update", i.e. shifted block causal. The inner objective descended by the update is
    the negative dot product `L = -<f(k), v>`. The trailing partial chunk is apply-only.

    Args:
        w0 (torch.Tensor):
            Fast weight of shape `[B, d_h, d_in]`.
        w1 (torch.Tensor):
            Fast weight of shape `[B, d_out, d_h]`.
        w2 (torch.Tensor):
            Fast weight of shape `[B, d_h, d_in]`.
        q (torch.Tensor):
            Queries of shape `[B, L, d_in]`.
        k (torch.Tensor):
            Keys of shape `[B, L, d_in]`.
        v (torch.Tensor):
            Values of shape `[B, L, d_out]`.
        lr0, lr1, lr2 (torch.Tensor):
            Per-token learning rates of shape `[B, L, 1]` (or `[B, L, d]`), one per fast weight.
        chunk_size (int):
            Number of tokens per test-time-training chunk. Note the update loop runs only when
            `L > chunk_size`; otherwise the fast weights are never updated. Default: `2048`.
        use_muon (bool):
            Whether to orthogonalize the fast-weight gradients via Newton-Schulz. Default: `False`.
        momentum (Optional[torch.Tensor]):
            Per-token momentum coefficients of shape `[B, L, 1]`, averaged within each chunk to a
            single scalar per chunk. If `None`, no momentum is applied. Default: `None`.
        prenorm (bool):
            If `True`, accumulate an unnormalized running state and use a renormalized copy for
            compute (`state = state + f(norm(state))`). If `False`, carry the renormalized weights
            forward directly (post-norm). Default: `False`.
        muon_group (str):
            `'per_matrix'` orthogonalizes `w0`, `w1` and `w2` independently — upstream LaCT's eager
            semantics and the scale-invariant choice. `'w0w2_joint'` orthogonalizes `w0` and `w2`
            stacked, matching what the fused Triton kernel does natively. Only used when
            `use_muon=True`. Default: `'per_matrix'`.

    Returns:
        o (torch.Tensor):
            Output of shape `[B, L, d_out]`.

    FLOPs:
        With `d_in == d_out == D` and SwiGLU hidden dim `H`, ignoring Muon and the trailing chunk:
        `4*D*H*L*B` (forward with key) + `8*D*H*L*B` (backward) + `6*D*H*L*B` (forward with query)
        = `18*D*H*L*B`.
    """
    if muon_group not in ('per_matrix', 'w0w2_joint'):
        raise ValueError(f"muon_group must be 'per_matrix' or 'w0w2_joint', got {muon_group!r}")

    # the activations drive the matmul precision; the fast weights may legitimately be fp32
    compute_dtype = q.dtype

    # target row norms the state is renormalized back onto after every update
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    # pre-norm keeps an unnormalized running state alongside the normalized one used for compute
    w0_main, w1_main, w2_main = w0, w1, w2

    if momentum is not None:
        dw0_momentum = torch.zeros_like(w0)
        dw1_momentum = torch.zeros_like(w1)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [B, d_in, L]
    v = v.transpose(1, 2)  # [B, d_out, L]
    output = torch.zeros_like(v)

    seq_len = k.shape[1]
    e_index = 0
    for s_index in range(0, seq_len - chunk_size, chunk_size):
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]            # [B, L_c, d_in]
        vi = v[:, :, s_index:e_index]            # [B, d_out, L_c]
        qi = q[:, :, s_index:e_index]            # [B, d_in, L_c]
        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]

        # ---- apply: read out with the fast weights as of the previous chunk ----
        h = _bmm(w2, qi, compute_dtype)
        gate = F.silu(_bmm(w0, qi, compute_dtype), inplace=True)
        output[:, :, s_index:e_index] = _bmm(w1, gate * h, compute_dtype)

        # ---- update: descend L = -<f(k), v> over the whole chunk at once ----
        kiT = ki.transpose(1, 2)
        gate_before_act = _bmm(w0, kiT, compute_dtype)
        hidden_before_mul = _bmm(w2, kiT, compute_dtype)
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        dhidden = _bmm(w1.transpose(1, 2), vi, compute_dtype)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = _bmm(vi, (hidden.transpose(1, 2) * lr1i).to(compute_dtype), compute_dtype)
        dw0 = _bmm(dgate_before_act, (ki * lr0i).to(compute_dtype), compute_dtype)
        dw2 = _bmm(dhidden_before_mul, (ki * lr2i).to(compute_dtype), compute_dtype)

        if momentum is not None:
            # one scalar coefficient per chunk per head
            m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True)
            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum, dw1_momentum, dw2_momentum = dw0, dw1, dw2

        if use_muon:
            dw0, dw1, dw2 = _apply_muon(dw0, dw1, dw2, muon_group)

        w0_main = w0_main + dw0
        w1_main = w1_main + dw1
        w2_main = w2_main + dw2

        # channel-wise L2 renorm back onto the initial row norms; conceptually a post-norm
        w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        if not prenorm:
            # post-norm carries the renormalized weights forward as the state itself
            w0_main, w1_main, w2_main = w0, w1, w2

    # trailing chunk: apply only, never update
    qi = q[:, :, e_index:seq_len]
    h = _bmm(w2, qi, compute_dtype)
    gate = F.silu(_bmm(w0, qi, compute_dtype), inplace=True)
    output[:, :, e_index:seq_len] = _bmm(w1, gate * h, compute_dtype)

    return output.transpose(1, 2)
