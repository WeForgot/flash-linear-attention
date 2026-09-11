
from __future__ import annotations

import warnings

from transformers.configuration_utils import PretrainedConfig


class LaCTConfig(PretrainedConfig):
    """
    Configuration for LaCT (Large-Chunk Test-Time Training), from "Test-Time Training Done Right".

    Each block mixes an in-layer sliding-window attention with a test-time-trained SwiGLU fast
    weight, sharing one QKV projection; the two outputs are summed before a single output projection.

    Args:
        hidden_size (int):
            Model dimension. Default: `2048`.
        num_hidden_layers (int):
            Number of blocks. Default: `24`.
        num_heads (int):
            Number of attention heads. Default: `32`.
        num_lact_heads (int):
            Number of fast-weight heads, deliberately far fewer than `num_heads`. Default: `4`.
        inter_multi (float):
            SwiGLU hidden multiplier for the fast weight. Default: `1`.
        qkv_bias (bool):
            Whether the shared QKV projection has a bias. Default: `False`.
        attn_qk_norm (bool):
            Whether to RMS-normalize q and k across the full hidden size. Default: `False`.
        lact_chunk_size (int):
            Tokens per test-time-training chunk. No fast-weight update occurs for sequences no
            longer than this, so small-context smoke tests should lower it. Default: `2048`.
        use_muon (bool):
            Whether to orthogonalize fast-weight gradients via Newton-Schulz. Default: `False`.
        muon_group (str):
            `'per_matrix'` or `'w0w2_joint'`. Only used when `use_muon=True`. Default: `'per_matrix'`.
        lr_dim (int):
            Per-head learning-rate channels. Default: `1`.
        qkv_silu (bool):
            Whether to apply SiLU to the fast-weight q/k/v. Default: `True`.
        no_v_silu (bool):
            If `True`, skip SiLU on the fast-weight v even when `qkv_silu` is set. Default: `False`.
        lr_parameterization (str):
            Only `'mamba'` is supported. Default: `'mamba'`.
        learnable_ttt_scale (bool):
            Whether to gate the fast-weight output with a learned per-head scale. Default: `True`.
        use_momentum (bool):
            Whether to apply per-chunk momentum to the fast-weight update. Default: `True`.
        ttt_loss_type (str):
            Inner objective; only `'dot_product'` is supported. Default: `'dot_product'`.
        ttt_prenorm (bool):
            `True` for `state = state + f(norm(state))`, `False` for `state = norm(state + f(state))`.
            Default: `True`.
        ttt_nope (bool):
            If `True`, no positional encoding on the fast-weight q/k. Default: `False`.
        w0_w2_low_rank (int):
            `-1` keeps the initial fast weights dense; a positive value sets the factorization rank.
            Default: `-1`.
        window_size (Optional[int]):
            Sliding-window attention span in tokens including self; `None` is full causal.
            Default: `2048`.
        rope_theta (float):
            RoPE base. Default: `10000.`.
        max_position_embeddings (int):
            Default: `2048`.
        fw_init_gain (float):
            Initialization gain for the fast weights. Default: `0.5`.
        use_fused_kernel (bool):
            Whether to use the fused Triton LaCT operator, which requires bfloat16 activations.
            Default: `False`.
        fp32_states (bool):
            Whether to accumulate the fast weights in fp32. Default: `False`.
    """

    model_type = 'lact'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_lact_heads: int = 4,
        inter_multi: float = 1,
        qkv_bias: bool = False,
        attn_qk_norm: bool = False,
        lact_chunk_size: int = 2048,
        use_muon: bool = False,
        muon_group: str = 'per_matrix',
        lr_dim: int = 1,
        qkv_silu: bool = True,
        no_v_silu: bool = False,
        lr_parameterization: str = 'mamba',
        learnable_ttt_scale: bool = True,
        use_momentum: bool = True,
        ttt_loss_type: str = 'dot_product',
        ttt_prenorm: bool = True,
        ttt_nope: bool = False,
        w0_w2_low_rank: int = -1,
        window_size: int | None = 2048,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.006,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        last_layer_fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        fw_init_gain: float = 0.5,
        use_fused_kernel: bool = False,
        fp32_states: bool = False,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_lact_heads = num_lact_heads
        self.inter_multi = inter_multi
        self.qkv_bias = qkv_bias
        self.attn_qk_norm = attn_qk_norm
        self.lact_chunk_size = lact_chunk_size
        self.use_muon = use_muon
        self.muon_group = muon_group
        self.lr_dim = lr_dim
        self.qkv_silu = qkv_silu
        self.no_v_silu = no_v_silu
        self.window_size = window_size
        self.lr_parameterization = lr_parameterization
        self.learnable_ttt_scale = learnable_ttt_scale
        self.ttt_prenorm = ttt_prenorm
        self.ttt_nope = ttt_nope
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        # NOTE: set False to use activation checkpointing on every layer
        self.last_layer_fuse_norm = last_layer_fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        self.use_momentum = use_momentum
        self.ttt_loss_type = ttt_loss_type
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain
        self.use_fused_kernel = use_fused_kernel
        self.fp32_states = fp32_states

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time."
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improve memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting."
            )
        if muon_group not in ('per_matrix', 'w0w2_joint'):
            raise ValueError(f"muon_group must be 'per_matrix' or 'w0w2_joint', got {muon_group!r}")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
