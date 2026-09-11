
import pytest
import torch
import torch.nn.functional as F

from fla.ops.attn import parallel_attn
from fla.ops.lact import chunk_lact_swiglu, fused_chunk_lact_swiglu, naive_lact_swiglu
from fla.utils import assert_close, device


def make_inputs(B, T, D, DH, dtype, seed=42, requires_grad=False):
    torch.manual_seed(seed)
    w0 = (torch.randn(B, DH, D, device=device) / D**0.5).to(dtype)
    w2 = (torch.randn(B, DH, D, device=device) / D**0.5).to(dtype)
    w1 = (torch.randn(B, D, DH, device=device) / DH**0.5).to(dtype)
    q = torch.randn(B, T, D, device=device, dtype=dtype)
    k = torch.randn(B, T, D, device=device, dtype=dtype)
    v = torch.randn(B, T, D, device=device, dtype=dtype)
    lr = [torch.rand(B, T, 1, device=device, dtype=torch.float32) * 1e-2 for _ in range(3)]
    momentum = torch.rand(B, T, 1, device=device, dtype=torch.float32)
    tensors = [w0, w1, w2, q, k, v]
    if requires_grad:
        tensors = [t.clone().requires_grad_(True) for t in tensors]
    return (*tensors, *lr, momentum)


# ===================================================================================
# Eager path
# ===================================================================================
@pytest.mark.parametrize(
    ['B', 'T', 'D', 'DH', 'C', 'prenorm', 'use_muon', 'muon_group', 'use_momentum', 'dtype'],
    [
        pytest.param(*t, id="B{}-T{}-D{}-DH{}-C{}-pre{}-muon{}-{}-mom{}-{}".format(*t))
        for t in [
            (2, 300, 32, 48, 64, False, False, 'per_matrix', False, torch.float32),
            (2, 300, 32, 48, 64, True, False, 'per_matrix', True, torch.float32),
            (2, 300, 32, 48, 64, True, True, 'per_matrix', True, torch.float32),
            (2, 300, 32, 48, 64, False, True, 'w0w2_joint', True, torch.float32),
            (1, 129, 16, 16, 64, True, False, 'per_matrix', False, torch.float32),
        ]
    ],
)
def test_chunk(B, T, D, DH, C, prenorm, use_muon, muon_group, use_momentum, dtype):
    """`chunk_lact_swiglu` must agree with the naive reference, forward and backward."""
    *t, lr0, lr1, lr2, mom = make_inputs(B, T, D, DH, dtype, requires_grad=True)
    w0, w1, w2, q, k, v = t
    momentum = mom if use_momentum else None
    kwargs = dict(chunk_size=C, use_muon=use_muon, momentum=momentum,
                  prenorm=prenorm, muon_group=muon_group)

    ref = naive_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, **kwargs)
    ref.sum().backward()
    ref_grads = [x.grad.clone() for x in t]
    for x in t:
        x.grad = None

    tri = chunk_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, **kwargs)
    tri.sum().backward()
    tri_grads = [x.grad.clone() for x in t]

    assert_close('o', ref, tri, 1e-5)
    for name, g_ref, g_tri in zip(['w0', 'w1', 'w2', 'q', 'k', 'v'], ref_grads, tri_grads):
        assert_close(f'd{name}', g_ref, g_tri, 1e-5)


def test_chunk_no_update_below_chunk_size():
    """With T <= chunk_size the fast weights never update; the op degenerates to a fixed MLP."""
    *t, lr0, lr1, lr2, _ = make_inputs(2, 64, 32, 48, torch.float32)
    w0, w1, w2, q, k, v = t
    small = chunk_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, chunk_size=64)
    large = chunk_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, chunk_size=4096)
    assert_close('o', small, large, 1e-6)


def test_chunk_rejects_cu_seqlens():
    """Packing must fail loudly: LaCT's chunks would otherwise leak state across documents."""
    *t, lr0, lr1, lr2, _ = make_inputs(1, 128, 32, 48, torch.float32)
    w0, w1, w2, q, k, v = t
    with pytest.raises(NotImplementedError, match='cu_seqlens'):
        chunk_lact_swiglu(
            w0, w1, w2, q, k, v, lr0, lr1, lr2, chunk_size=64,
            cu_seqlens=torch.tensor([0, 64, 128], dtype=torch.int32, device=device),
        )


def test_muon_groups_differ():
    """The two Muon groupings are genuinely different updates, not a reparameterization."""
    *t, lr0, lr1, lr2, _ = make_inputs(2, 300, 32, 32, torch.float32)
    w0, w1, w2, q, k, v = t
    kw = dict(chunk_size=64, use_muon=True)
    a = naive_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, muon_group='per_matrix', **kw)
    b = naive_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, muon_group='w0w2_joint', **kw)
    assert not torch.allclose(a, b, atol=1e-3), \
        "per_matrix and w0w2_joint should not agree; if they do, the grouping is not wired through"


# ===================================================================================
# Fused Triton path
# ===================================================================================
@pytest.mark.parametrize(
    ['B', 'T', 'D', 'DH', 'C', 'prenorm', 'use_muon', 'use_momentum'],
    [
        pytest.param(*t, id="B{}-T{}-D{}-DH{}-C{}-pre{}-muon{}-mom{}".format(*t))
        for t in [
            (2, 256, 64, 64, 64, False, False, False),
            (2, 256, 64, 64, 64, True, False, True),
            (2, 256, 64, 64, 64, True, True, True),
        ]
    ],
)
def test_fused_chunk(B, T, D, DH, C, prenorm, use_muon, use_momentum):
    """The fused Triton path must track the naive reference within bf16 tolerance."""
    *t, lr0, lr1, lr2, mom = make_inputs(B, T, D, DH, torch.bfloat16)
    w0, w1, w2, q, k, v = t
    momentum = mom if use_momentum else None
    kwargs = dict(chunk_size=C, use_muon=use_muon, momentum=momentum,
                  prenorm=prenorm, muon_group='per_matrix')

    ref = naive_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, **kwargs)
    tri = fused_chunk_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, **kwargs)
    # fused epilogues accumulate in fp32 where the eager path rounds to bf16 between steps, so the
    # tolerance is looser than an eager-vs-eager comparison would warrant
    assert_close('o', ref, tri, 8e-2)


def test_fused_chunk_requires_bf16():
    """The fused kernels assert bfloat16; the failure should be a clear error, not a kernel abort."""
    *t, lr0, lr1, lr2, _ = make_inputs(2, 256, 64, 64, torch.float32)
    w0, w1, w2, q, k, v = t
    with pytest.raises(TypeError, match='bfloat16'):
        fused_chunk_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, chunk_size=64)


def _reference_swa(q, k, v, window_size):
    """Causal (optionally windowed) softmax attention, written straight from the definition."""
    B, T, H, D = q.shape
    qs, ks, vs = (x.transpose(1, 2).float() for x in (q, k, v))
    scores = (qs @ ks.transpose(-1, -2)) / (D ** 0.5)
    i = torch.arange(T, device=q.device)[:, None]
    j = torch.arange(T, device=q.device)[None, :]
    allowed = j <= i
    if window_size is not None:
        allowed = allowed & ((i - j) < window_size)
    return (F.softmax(scores.masked_fill(~allowed, float('-inf')), dim=-1) @ vs).transpose(1, 2)


@pytest.mark.parametrize(
    ['T', 'W'],
    [(256, 64), (256, 128), (256, None), (256, 512), (129, 32), (64, 64)],
    ids=['T256-W64', 'T256-W128', 'T256-full', 'T256-W>T', 'T129-W32', 'T64-W=T'],
)
def test_sliding_window_matches_flash_attn_semantics(T, W):
    """
    The port replaced `flash_attn_func(causal=True, window_size=(W-1, 0))` with
    `parallel_attn(window_size=W)`. Both are defined as "W keys including self", so the off-by-one
    cancels. `flash_attn` needs sm_80+ and cannot run on every dev box, so this pins the semantics
    against the definition rather than against the other implementation.
    """
    B, H, D = 2, 4, 64
    q, k, v = (torch.randn(B, T, H, D, device=device, dtype=torch.float32) for _ in range(3))
    assert_close('o', _reference_swa(q, k, v, W), parallel_attn(q, k, v, window_size=W), 1e-4)


def test_fused_chunk_backward():
    """Gradients must flow through the fused kernels' hand-written backward."""
    *t, lr0, lr1, lr2, mom = make_inputs(2, 256, 64, 64, torch.bfloat16, requires_grad=True)
    w0, w1, w2, q, k, v = t
    o = fused_chunk_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2,
                                chunk_size=64, momentum=mom, prenorm=True)
    o.float().sum().backward()
    for name, x in zip(['w0', 'w1', 'w2', 'q', 'k', 'v'], t):
        assert x.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(x.grad).all(), f"{name} gradient is not finite"
