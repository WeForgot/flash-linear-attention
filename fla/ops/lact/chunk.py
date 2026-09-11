
"""
Eager (non-fused) LaCT SwiGLU operator.

This is the public entry point corresponding to upstream LaCT's `ttt_operation.py`. It shares its
chunk loop with `naive.py` — the two are the same algorithm, and that is deliberate: LaCT ships
exactly two implementations, an eager one and a fused-Triton one, and this is the eager one made to
follow FLA's op conventions. The fused path in `fused_chunk.py` is what a test against the naive
reference actually exercises.

Two deviations from upstream, both intentional (see the port plan):
  * no `@torch.compile` — FLA's convention is `@torch.compiler.disable` on a public op, so a
    trainer's outer compile treats it as an opaque leaf;
  * no hard `@torch.autocast(bfloat16)` — the op follows the caller's dtype and ambient autocast,
    which is what makes an fp32 reference test meaningful.
"""

from __future__ import annotations

import torch

from fla.ops.lact.naive import naive_lact_swiglu
from fla.utils import input_guard

__all__ = ['chunk_lact_swiglu']


@torch.compiler.disable
@input_guard
def chunk_lact_swiglu(
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
    Block-causal LaCT with a SwiGLU fast-weight function, `f(x) = w1 @ (silu(w0 @ x) * (w2 @ x))`.

    Each chunk is applied with the previous chunk's fast weights and only then used to update them
    ("apply then update"), which makes the operator strictly causal at chunk granularity. The inner
    objective is the negative dot product `L = -<f(k), v>`. The trailing partial chunk is apply-only.

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
            Per-token learning rates of shape `[B, L, 1]`, one per fast weight.
        chunk_size (int):
            Tokens per test-time-training chunk. The update loop runs only when `L > chunk_size`;
            below that the fast weights are never updated and the op degenerates to a fixed MLP.
            Default: `2048`.
        use_muon (bool):
            Whether to orthogonalize the fast-weight gradients via Newton-Schulz. Default: `False`.
        momentum (Optional[torch.Tensor]):
            Per-token momentum coefficients of shape `[B, L, 1]`, averaged to one scalar per chunk.
            Default: `None`.
        prenorm (bool):
            If `True`, accumulate an unnormalized running state and compute with a renormalized copy.
            If `False`, carry the renormalized weights forward (post-norm). Default: `False`.
        muon_group (str):
            `'per_matrix'` orthogonalizes `w0`/`w1`/`w2` independently; `'w0w2_joint'` orthogonalizes
            `w0` and `w2` stacked, matching the fused kernel. Default: `'per_matrix'`.

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
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError(
            f"q, k and v must share a sequence length, got {q.shape[1]}, {k.shape[1]}, {v.shape[1]}."
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    return naive_lact_swiglu(
        w0=w0, w1=w1, w2=w2,
        q=q, k=k, v=v,
        lr0=lr0, lr1=lr1, lr2=lr2,
        chunk_size=chunk_size,
        use_muon=use_muon,
        momentum=momentum,
        prenorm=prenorm,
        muon_group=muon_group,
    )
