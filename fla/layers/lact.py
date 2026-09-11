
"""
LaCT: Large-Chunk Test-Time Training, from "Test-Time Training Done Right"
(https://arxiv.org/abs/2505.23884).

Ported from the reference release (`lact_model/layer_lact_swiglu.py`) with one substantive change:
the in-layer sliding-window attention runs on FLA's Triton attention (`fla.ops.attn`) instead of the
`flash_attn` package. The window semantics are identical — FlashAttention-2's
`window_size=(W - 1, 0)` with `causal=True` and FLA's `window_size=W` both mean "W keys including
self" — but this build has no `flash_attn` dependency and runs on pre-Ampere cards.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.attn.parallel import parallel_attn
from fla.ops.lact import chunk_lact_swiglu, fused_chunk_lact_swiglu

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)

__all__ = ['LaCT', 'LowRankFastWeight']


def inv_softplus(x):
    """Inverse of softplus, used so the per-token learning rate starts at `base_lr`."""
    if isinstance(x, torch.Tensor):
        return x + torch.log(-torch.expm1(-x))
    return x + math.log(-math.expm1(-x))


class LowRankFastWeight(nn.Module):
    r"""
    Low-rank parameterization of an initial fast weight, `W = W_left @ W_right (+ 0.5 * I)`.

    This exists to keep the parameter count comparable against baselines; upstream notes that the
    low-rank form, ideally, always hurts quality relative to a dense initial fast weight.

    Args:
        num_heads (int):
            Number of fast-weight heads.
        out_features (int):
            Output dimension of the fast weight.
        in_features (int):
            Input dimension of the fast weight.
        rank (int):
            Rank of the factorization. Default: `32`.
        init_gain (float):
            Scales the initialization std of both factors. Default: `0.5`.
        add_identity (bool):
            Whether to add `0.5 * I` to the product. Default: `False`.
    """

    def __init__(
        self,
        num_heads: int,
        out_features: int,
        in_features: int,
        rank: int = 32,
        init_gain: float = 0.5,
        add_identity: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.add_identity = add_identity
        self.init_gain = init_gain

        self.w_left = nn.Parameter(torch.randn(num_heads, out_features, rank))
        self.w_right = nn.Parameter(torch.randn(num_heads, rank, in_features))

    def _init_weights(self):
        nn.init.normal_(self.w_left, std=1.0 / math.sqrt(self.rank) * self.init_gain)
        nn.init.normal_(self.w_right, std=1.0 / math.sqrt(self.in_features) * self.init_gain)

    def forward(self) -> torch.Tensor:
        W = self.w_left @ self.w_right
        if self.add_identity:
            eye = torch.eye(self.out_features, self.in_features, device=W.device, dtype=W.dtype)
            W = W + eye.unsqueeze(0) * 0.5
        return W


class LaCT(nn.Module):
    r"""
    LaCT-SwiGLU token mixer: sliding-window attention and a test-time-trained SwiGLU fast weight,
    sharing one QKV projection and summed before a single output projection (GAU-style).

    The fast weight `f(x) = w1 @ (silu(w0 @ x) * (w2 @ x))` is updated once per `lact_chunk_size`
    tokens by descending the inner loss `L = -<f(k), v>`. Each chunk is applied with the previous
    chunk's weights before contributing its own update, which keeps the operator strictly causal at
    chunk granularity. Attention supplies exact local recency; the fast weight supplies compressed
    long-range memory.

    Args:
        hidden_size (int):
            Model dimension. Default: `2048`.
        num_heads (int):
            Number of attention heads. The attention head dim is `hidden_size // num_heads`.
            Default: `32`.
        num_lact_heads (int):
            Number of fast-weight heads. Deliberately much smaller than `num_heads` so the
            fast-weight matmuls stay large and GPU-efficient. Default: `4`.
        inter_multi (float):
            SwiGLU hidden multiplier for the fast weight, `d_h = d_in * inter_multi`. Default: `1`.
        window_size (Optional[int]):
            Sliding-window span for attention, in tokens including self. `None` means full causal.
            Default: `2048`.
        lact_chunk_size (int):
            Tokens per test-time-training chunk. Note no fast-weight update happens at all when the
            sequence is no longer than this. Default: `2048`.
        qkv_bias (bool):
            Whether the shared QKV projection has a bias. Default: `False`.
        attn_qk_norm (bool):
            Whether to RMS-normalize q and k. Note this normalizes across the *full* hidden size
            rather than per head, which is intentional and differs from `fla.layers.Attention`.
            Default: `False`.
        qkv_silu (bool):
            Whether to apply SiLU to the fast-weight q/k/v. Default: `True`.
        no_v_silu (bool):
            If `True`, skip SiLU on the fast-weight v even when `qkv_silu` is set. Default: `False`.
        lr_dim (int):
            Per-head learning-rate channels; `1` gives a scalar LR per head. Default: `1`.
        use_muon (bool):
            Whether to orthogonalize fast-weight gradients via Newton-Schulz. Default: `False`.
        muon_group (str):
            `'per_matrix'` orthogonalizes `w0`/`w1`/`w2` independently; `'w0w2_joint'` orthogonalizes
            `w0` and `w2` stacked. Only used when `use_muon=True`. Default: `'per_matrix'`.
        lr_parameterization (str):
            How the per-token LR is produced. Only `'mamba'` (softplus) is supported.
            Default: `'mamba'`.
        learnable_ttt_scale (bool):
            Whether to apply a learned per-head gate to the fast-weight output. Default: `False`.
        ttt_prenorm (bool):
            `True` keeps an unnormalized running state and computes with a renormalized copy;
            `False` carries the renormalized weights forward. Default: `False`.
        ttt_nope (bool):
            If `True`, skip rotary embedding on the fast-weight q/k. Default: `False`.
        rope_theta (float):
            RoPE base. Default: `500000.`.
        layer_idx (int):
            Index of this layer, required when a cache is used. Default: `None`.
        max_position_embeddings (int):
            Default: `2048`.
        w0_w2_low_rank (int):
            Rank for the low-rank `w0`/`w2` parameterization; `-1` keeps them dense. Default: `-1`.
        use_momentum (bool):
            Whether to apply per-chunk momentum to the fast-weight update. Default: `False`.
        ttt_loss_type (str):
            Inner objective. Only `'dot_product'` is supported. Default: `'dot_product'`.
        fw_init_gain (float):
            Initialization gain for the fast weights. Default: `0.5`.
        use_fused_kernel (bool):
            Whether to use the fused Triton LaCT operator. Requires bfloat16 activations.
            Default: `False`.
        fp32_states (bool):
            Whether to keep the fast weights in fp32 while the matmuls stay in the activation dtype.
            Default: `False`.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_lact_heads: int = 4,
        inter_multi: float = 1,
        window_size: int | None = 2048,
        lact_chunk_size: int = 2048,
        qkv_bias: bool = False,
        attn_qk_norm: bool = False,
        qkv_silu: bool = True,
        no_v_silu: bool = False,
        lr_dim: int = 1,
        use_muon: bool = False,
        muon_group: str = 'per_matrix',
        lr_parameterization: str = "mamba",
        learnable_ttt_scale: bool = False,
        ttt_prenorm: bool = False,
        ttt_nope: bool = False,
        rope_theta: float = 500000.,
        layer_idx: int = None,
        max_position_embeddings: int = 2048,
        w0_w2_low_rank: int = -1,
        use_momentum: bool = False,
        ttt_loss_type: str = "dot_product",
        fw_init_gain: float = 0.5,
        use_fused_kernel: bool = False,
        fp32_states: bool = False,
        **kwargs,
    ) -> LaCT:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        if hidden_size % num_lact_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_lact_heads ({num_lact_heads})."
            )
        if ttt_loss_type != "dot_product":
            raise NotImplementedError(f"ttt_loss_type {ttt_loss_type!r} is not supported.")
        if lr_parameterization.lower() != "mamba":
            raise NotImplementedError(
                f"lr_parameterization {lr_parameterization!r} is not supported."
            )
        if muon_group not in ('per_matrix', 'w0w2_joint'):
            raise ValueError(f"muon_group must be 'per_matrix' or 'w0w2_joint', got {muon_group!r}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.inter_multi = inter_multi
        self.window_size = window_size
        self.head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.attn_qk_norm = attn_qk_norm
        if self.attn_qk_norm:
            # NOTE: deliberately over the full hidden size, not per head. One shared RMS statistic
            # across heads is what upstream LaCT does, and the result feeds the fast-weight branch
            # too (via `_rescale_qk`), so making it per-head would change TTT as well as attention.
            self.q_norm = RMSNorm(self.hidden_size)
            self.k_norm = RMSNorm(self.hidden_size)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

        # ---- fast weights ----
        self.use_muon = use_muon
        self.muon_group = muon_group
        self.lact_chunk_size = lact_chunk_size
        self.num_fw_heads = num_lact_heads
        self.fw_head_dim = self.hidden_size // self.num_fw_heads
        self.qkv_silu = qkv_silu
        self.no_v_silu = no_v_silu
        self.ttt_prenorm = ttt_prenorm
        self.ttt_nope = ttt_nope

        d_in = d_out = self.fw_head_dim
        d_h = int(d_in * inter_multi)
        self.d_in, self.d_out, self.d_h = d_in, d_out, d_h
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain

        if self.w0_w2_low_rank > 0:
            self.w0 = LowRankFastWeight(
                self.num_fw_heads, d_h, d_in, self.w0_w2_low_rank,
                init_gain=self.fw_init_gain, add_identity=True,
            )
            self.w2 = LowRankFastWeight(
                self.num_fw_heads, d_h, d_in, self.w0_w2_low_rank,
                init_gain=self.fw_init_gain, add_identity=True,
            )
        else:
            self.w0 = nn.Parameter(torch.randn(self.num_fw_heads, d_h, d_in) / math.sqrt(d_in))
            self.w2 = nn.Parameter(torch.randn(self.num_fw_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(torch.randn(self.num_fw_heads, d_out, d_h) / math.sqrt(d_h))

        # ---- per-token learning rate ----
        self.lr_dim = int(lr_dim * 3 * self.num_fw_heads)
        self.lr_proj = nn.Linear(self.hidden_size, self.lr_dim)
        self.lr_parameterization = lr_parameterization
        self.base_lr_inv = inv_softplus(0.001)

        # ---- per-channel affine separating the TTT q/k from the attention q/k ----
        self.qk_scale = nn.Parameter(torch.ones(hidden_size, 2))
        self.qk_offset = nn.Parameter(torch.zeros(hidden_size, 2))

        self.learnable_ttt_scale = learnable_ttt_scale
        if self.learnable_ttt_scale:
            self.ttt_scale_proj = nn.Linear(hidden_size, self.num_fw_heads)

        # NOTE: one gain shared across fast-weight heads. FLA's per-head idiom would be
        # `GroupNorm(num_groups=num_lact_heads, ..., is_rms_norm=True)`, which has a different
        # parameter shape -- not a free swap. Kept as upstream wrote it.
        self.ttt_norm = RMSNorm(self.fw_head_dim, elementwise_affine=True)

        self.use_momentum = use_momentum
        if self.use_momentum:
            self.momentum_proj = nn.Sequential(
                nn.Linear(hidden_size, self.num_fw_heads),
                nn.Sigmoid(),
            )

        self.ttt_loss_type = ttt_loss_type
        self.use_fused_kernel = use_fused_kernel
        self.fp32_states = fp32_states

    def _rescale_qk(self, q: torch.Tensor, k: torch.Tensor):
        """Per-channel affine that forks the fast-weight q/k away from the attention q/k."""
        qk_scale = self.qk_scale.view(1, 1, -1, 2)
        qk_offset = self.qk_offset.view(1, 1, -1, 2)
        q = q * qk_scale[:, :, :, 0] + qk_offset[:, :, :, 0]
        k = k * qk_scale[:, :, :, 1] + qk_offset[:, :, :, 1]
        return q, k

    def _lact(self, w0, w1, w2, q, k, v, lr0, lr1, lr2, momentum):
        """Dispatch to the fused Triton or eager LaCT operator."""
        op = fused_chunk_lact_swiglu if self.use_fused_kernel else chunk_lact_swiglu
        return op(
            w0, w1, w2, q, k, v, lr0, lr1, lr2,
            chunk_size=self.lact_chunk_size,
            use_muon=self.use_muon,
            momentum=momentum,
            prenorm=self.ttt_prenorm,
            muon_group=self.muon_group,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()
        cu_seqlens = kwargs.get('cu_seqlens')

        # The fast-weight chunks are not document-aware, so packing would carry state across
        # document boundaries. Refuse rather than silently train on leaked context.
        if cu_seqlens is not None:
            raise NotImplementedError(
                "LaCT does not support `cu_seqlens`: its test-time-training chunks stride the "
                "packed stream without regard for document boundaries, so fast-weight state (and "
                "momentum) would leak across documents. Use one document per sequence."
            )
        # Padding is only unsafe once a fast-weight update can actually see a pad token, which
        # requires the sequence to be longer than one chunk.
        if attention_mask is not None and q_len > self.lact_chunk_size:
            if (attention_mask == 0).any():
                raise NotImplementedError(
                    f"LaCT received a padded batch with q_len ({q_len}) greater than "
                    f"lact_chunk_size ({self.lact_chunk_size}); padding tokens would enter a "
                    f"fast-weight update. Use unpadded batches at this sequence length."
                )

        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        if self.attn_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # fork the test-time-training branch off before the attention reshape
        fast_q, fast_k = self._rescale_qk(q, k)
        fast_v = v

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset
            if attention_mask is not None:
                seqlen_offset = seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)
        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                # the cache is rolled to the window, which is what makes windowed decoding correct
                cache_kwargs=dict(window_size=self.window_size),
            )["attn_state"]
            if cache_has_content:
                k = rearrange(k_cached, "... (h d) -> ... h d", d=self.head_dim)
                v = rearrange(v_cached, "... (h d) -> ... h d", d=self.head_dim)

        # ---- sliding-window causal attention, on Triton ----
        indices_q = None
        if attention_mask is not None:
            q, (k, v), indices_q, cu_seq_lens, max_seq_lens = unpad_input(
                q, (k, v), attention_mask, q_len, keepdim=True
            )
            _, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            if max_seqlen_q != max_seqlen_k:
                # single-token decode; the cache above is already truncated to the window
                o = attn_decoding_one_step(q, k, v, cu_seqlens=cu_seqlens_k)
            else:
                o = parallel_attn(q, k, v, window_size=self.window_size, cu_seqlens=cu_seqlens_k)
        else:
            o = parallel_attn(q, k, v, window_size=self.window_size)
        if indices_q is not None:
            o = pad_input(o.squeeze(0), indices_q, batch_size, q_len)
        o = o.reshape(batch_size, q_len, -1)

        # ---- test-time training ----
        fast_q, fast_k, fast_v = (
            rearrange(x, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)
            for x in (fast_q, fast_k, fast_v)
        )

        if self.qkv_silu:
            fast_q = F.silu(fast_q)
            fast_k = F.silu(fast_k)
            if not self.no_v_silu:
                fast_v = F.silu(fast_v)

        fast_q = _l2_norm(fast_q)
        fast_k = _l2_norm(fast_k)

        if not self.ttt_nope:
            # reuse the attention RoPE: fold back to [b, s, d], split by *attention* heads so the
            # rotary sees its own head_dim, then re-split by fast-weight heads
            fast_q, fast_k = (
                rearrange(x, "(b n_h) s d -> b s (n_h d)", n_h=self.num_fw_heads)
                for x in (fast_q, fast_k)
            )
            fast_q, fast_k = (
                rearrange(x, "b s (n_h d) -> b s n_h d", n_h=self.num_heads)
                for x in (fast_q, fast_k)
            )
            fast_q, fast_k = self.rotary(
                fast_q, fast_k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen
            )
            fast_q, fast_k = (
                rearrange(x, "b s n_h d -> b s (n_h d)", n_h=self.num_heads)
                for x in (fast_q, fast_k)
            )
            fast_q, fast_k = (
                rearrange(x, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)
                for x in (fast_q, fast_k)
            )

        if self.w0_w2_low_rank > 0:
            fw_w0 = self.w0().repeat(batch_size, 1, 1)
            fw_w2 = self.w2().repeat(batch_size, 1, 1)
        else:
            fw_w0 = self.w0.repeat(batch_size, 1, 1)
            fw_w2 = self.w2.repeat(batch_size, 1, 1)
        fw_w1 = self.w1.repeat(batch_size, 1, 1)

        lr = self.lr_proj(hidden_states)
        lr = F.softplus(lr.float() + self.base_lr_inv)
        fw_lr = rearrange(lr, "b s (n_h lr_dim) -> (b n_h) s lr_dim", n_h=self.num_fw_heads)
        fw_lr0, fw_lr1, fw_lr2 = fw_lr.chunk(3, dim=-1)

        momentum = None
        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states).float()
            momentum = rearrange(momentum, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)

        if self.fp32_states:
            # the fast weights accumulate in fp32 while the matmuls stay in the activation dtype,
            # mirroring how bf16 training keeps fp32 slow weights
            fw_w0 = fw_w0.to(torch.float32)
            fw_w1 = fw_w1.to(torch.float32)
            fw_w2 = fw_w2.to(torch.float32)

        fw_x = self._lact(fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v,
                          fw_lr0, fw_lr1, fw_lr2, momentum)

        ttt_x_normed = self.ttt_norm(fw_x)
        if self.learnable_ttt_scale:
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            ttt_scale = rearrange(ttt_scale, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)
            ttt_x_normed = ttt_x_normed * ttt_scale
        ttt_x_normed = rearrange(ttt_x_normed, "(b n_h) s d -> b s (n_h d)", n_h=self.num_fw_heads)

        # attention and the fast weight are summed, then share one output projection
        o = self.o_proj(o + ttt_x_normed)

        return o, None, past_key_values


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 norm with eps outside the root, matching LaCT's fast-weight normalization."""
    dtype = x.dtype
    return (x / (x.norm(dim=-1, keepdim=True) + 1e-5)).to(dtype)
