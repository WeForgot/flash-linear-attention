
"""
Fused-Triton LaCT SwiGLU operator.

Ported from upstream LaCT's `ttt_operation_fused_kernel.py`. Relative to the eager path in
`chunk.py` this:

  * concatenates `w0` and `w2` into a single `[B, 2*d_h, d_in]` buffer so the SwiGLU GEMMs share one
    operand and one epilogue;
  * zero-pads the sequence up to a multiple of `chunk_size` and reshapes to `[n, B, c, d]` instead of
    slicing, so every chunk is contiguous;
  * fuses the weight update and the channel-wise L2 renormalization into `l2_norm_add_fused`.

The Triton kernels require bfloat16 activations, so unlike `chunk_lact_swiglu` this path is not
dtype-transparent. It is the HPCC/H200 path; on older cards prefer `chunk_lact_swiglu`.

Upstream applies Muon to the *concatenated* `[w0; w2]` block, which is a different update from
orthogonalizing each matrix separately. That choice is exposed here as `muon_group` so both the
eager and fused paths can implement the same algorithm and be tested against one reference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from fla.ops.lact.kernels import (
    fused_lact_swiglu_ffn_fast_weight_grads,
    fused_swiglu_ffn_fwd,
    l2_norm_add_fused,
)
from fla.ops.lact.naive import zeropower_via_newtonschulz5
from fla.utils import input_guard

__all__ = ['fused_chunk_lact_swiglu']


def _muon(dw0_w2: torch.Tensor, dw1: torch.Tensor, d_h: int, muon_group: str):
    """Orthogonalize the fast-weight gradients, honouring the requested w0/w2 grouping."""
    dw1 = zeropower_via_newtonschulz5(dw1)
    if muon_group == 'w0w2_joint':
        return zeropower_via_newtonschulz5(dw0_w2), dw1
    if muon_group == 'per_matrix':
        # split the fused buffer so each matrix is orthogonalized on its own, matching the eager path
        dw0 = zeropower_via_newtonschulz5(dw0_w2[:, :d_h].contiguous())
        dw2 = zeropower_via_newtonschulz5(dw0_w2[:, d_h:].contiguous())
        return torch.cat([dw0, dw2], dim=1), dw1
    raise ValueError(f"muon_group must be 'per_matrix' or 'w0w2_joint', got {muon_group!r}")


@torch.compiler.disable
@input_guard
def fused_chunk_lact_swiglu(
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
    **kwargs,
) -> torch.Tensor:
    r"""
    Block-causal LaCT with a SwiGLU fast-weight function, backed by fused Triton kernels.

    Numerically equivalent to `chunk_lact_swiglu` up to precision, with one exception: when
    `use_muon=True`, `muon_group` selects a genuinely different update rule, so both paths must be
    given the same value to agree.

    Args:
        w0 (torch.Tensor):
            Fast weight of shape `[B, d_h, d_in]`.
        w1 (torch.Tensor):
            Fast weight of shape `[B, d_out, d_h]`.
        w2 (torch.Tensor):
            Fast weight of shape `[B, d_h, d_in]`.
        q (torch.Tensor):
            Queries of shape `[B, L, d_in]`, bfloat16.
        k (torch.Tensor):
            Keys of shape `[B, L, d_in]`, bfloat16.
        v (torch.Tensor):
            Values of shape `[B, L, d_out]`, bfloat16.
        lr0, lr1, lr2 (torch.Tensor):
            Per-token learning rates of shape `[B, L, 1]`, one per fast weight.
        chunk_size (int):
            Tokens per test-time-training chunk. The update loop runs only when `L > chunk_size`.
            Default: `2048`.
        use_muon (bool):
            Whether to orthogonalize the fast-weight gradients via Newton-Schulz. Default: `False`.
        momentum (Optional[torch.Tensor]):
            Per-token momentum coefficients of shape `[B, L, 1]`, averaged to one scalar per chunk.
            Default: `None`.
        prenorm (bool):
            If `True`, accumulate an unnormalized running state and compute with a renormalized copy.
            If `False`, use the fused add-and-renormalize kernel (post-norm). Default: `False`.
        muon_group (str):
            `'per_matrix'` orthogonalizes `w0`/`w1`/`w2` independently; `'w0w2_joint'` orthogonalizes
            the concatenated `[w0; w2]` block, which is what upstream's fused path did
            unconditionally. Default: `'per_matrix'`.

    Returns:
        o (torch.Tensor):
            Output of shape `[B, L, d_out]`.
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if 'cu_seqlens' in kwargs and kwargs['cu_seqlens'] is not None:
        raise NotImplementedError(
            "LaCT's test-time-training chunks are not document-aware, so passing `cu_seqlens` would "
            "silently carry fast-weight state across document boundaries. Flatten to one document "
            "per sequence, or implement varlen-aware chunking first."
        )
    if muon_group not in ('per_matrix', 'w0w2_joint'):
        raise ValueError(f"muon_group must be 'per_matrix' or 'w0w2_joint', got {muon_group!r}")
    if q.dtype != torch.bfloat16:
        raise TypeError(
            f"The fused LaCT kernels require bfloat16 activations, got {q.dtype}. "
            f"Use `chunk_lact_swiglu` for other dtypes."
        )

    d_h = w0.shape[1]
    w0_w2 = torch.cat([w0, w2], dim=1).contiguous()
    # post-norm feeds the fused kernel, which wants per-row scales without a trailing singleton dim
    w0_w2_norm = w0_w2.norm(dim=2, keepdim=prenorm)
    w1_norm = w1.norm(dim=2, keepdim=prenorm)

    # pre-norm keeps a full-precision running state and computes against a bf16 normalized copy
    w0_w2_main, w1_main = w0_w2, w1
    if prenorm:
        w0_w2 = w0_w2.to(torch.bfloat16)
        w1 = w1.to(torch.bfloat16)

    if momentum is not None:
        dw0_dw2_momentum = torch.zeros_like(w0_w2_main)
        dw1_momentum = torch.zeros_like(w1_main)

    q_len = q.shape[1]
    pad = -q_len % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    lr0, lr1, lr2 = (F.pad(x, (0, 0, 0, pad)) for x in (lr0, lr1, lr2))
    if momentum is not None:
        momentum = F.pad(momentum, (0, 0, 0, pad))
    num_chunks = (q_len + pad) // chunk_size

    q, k, v = (rearrange(x, "b (n c) d -> n b c d", n=num_chunks) for x in (q, k, v))
    lr0, lr1, lr2 = (
        rearrange(x, "b (n c) d -> n b (c d)", n=num_chunks, d=1) for x in (lr0, lr1, lr2)
    )
    if momentum is not None:
        momentum = rearrange(momentum, "b (n c) 1 -> n b c 1", n=num_chunks)

    output = torch.zeros_like(q)

    for i in range(num_chunks - 1):
        ki, vi, qi = k[i].contiguous(), v[i].contiguous(), q[i].contiguous()
        lr0i, lr1i, lr2i = lr0[i].contiguous(), lr1[i].contiguous(), lr2[i].contiguous()

        w0_w2_bf16 = w0_w2.to(torch.bfloat16)
        w1_bf16 = w1.to(torch.bfloat16)

        # apply first, with the fast weights as of the previous chunk
        output[i] = fused_swiglu_ffn_fwd(w0_w2_bf16, w1_bf16, qi)

        # then update, descending the negative dot-product loss over the whole chunk
        dw0_w2, dw1 = fused_lact_swiglu_ffn_fast_weight_grads(
            w0_w2_bf16, w1_bf16, ki, vi, lr0i, lr1i, lr2i
        )

        if momentum is not None:
            m_i = momentum[i].contiguous().mean(dim=1, keepdim=True)
            dw0_w2 = dw0_w2 + dw0_dw2_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw0_dw2_momentum, dw1_momentum = dw0_w2, dw1

        if use_muon:
            dw0_w2, dw1 = _muon(dw0_w2, dw1, d_h, muon_group)

        if prenorm:
            w0_w2_main = w0_w2_main + dw0_w2
            w1_main = w1_main + dw1
            w0_w2 = (
                w0_w2_main / (w0_w2_main.norm(dim=2, keepdim=True) + 1e-5) * w0_w2_norm
            ).to(torch.bfloat16)
            w1 = (w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm).to(torch.bfloat16)
        else:
            # fused add-then-renormalize back onto the initial row norms
            w0_w2 = l2_norm_add_fused(w0_w2, dw0_w2, w0_w2_norm, eps=1e-5)
            w1 = l2_norm_add_fused(w1, dw1, w1_norm, eps=1e-5)

    # trailing chunk: apply only, never update
    output[-1] = fused_swiglu_ffn_fwd(
        w0_w2.to(torch.bfloat16), w1.to(torch.bfloat16), q[-1].contiguous()
    )

    output = rearrange(output, "n b c d -> b (n c) d")
    return output[:, :q_len]
