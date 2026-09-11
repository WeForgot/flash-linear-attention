
"""
Triton kernels backing the fused LaCT (Large-Chunk Test-Time Training) operator.

Ported from the reference LaCT release (`lact_model/lact_triton_kernels/`) with a near 1:1 file
mapping so that future re-syncs against upstream stay tractable. Only the module names and the
cross-kernel imports were changed.
"""

from .ffn import FusedSwiGLUFFNFwd, fused_swiglu_ffn_fwd, reference_swiglu_ffn_fwd
from .fused_matmul import (
    fused_four_mm_same_out_interface,
    fused_two_mm_same_out_interface,
    fused_two_mm_same_out_wT_x_triton,
    fused_two_mm_same_out_wT_xT_triton,
)
from .fw_grad import (
    FusedLactSwiGLUFFNBwd,
    fused_lact_swiglu_ffn_fast_weight_grads,
    reference_lact_swiglu_ffn_fast_weight_grads,
)
from .l2norm_add import L2NormAddFunction, l2_norm_add_fused, reference_l2_norm_add_fused
from .pointwise import triton_swiglu_bwd_bwd_fused_cat_inp_out
from .prenorm_momentum import (
    PrenormUpdateWithMomentumAndL2NormFunction,
    fused_prenorm_update_with_momentum_and_l2_norm,
    reference_l2_norm_add_fused_with_momentum,
)
from .swiglu_bwd import swiglu_backward_three_bmm_ref, swiglu_backward_three_bmm_triton
from .swiglu_bwd_lr import swiglu_backward_three_bmm_with_lr_triton
from .swiglu_fwd import fused_two_mm_swiglu_triton

__all__ = [
    'FusedLactSwiGLUFFNBwd',
    'FusedSwiGLUFFNFwd',
    'L2NormAddFunction',
    'PrenormUpdateWithMomentumAndL2NormFunction',
    'fused_four_mm_same_out_interface',
    'fused_lact_swiglu_ffn_fast_weight_grads',
    'fused_prenorm_update_with_momentum_and_l2_norm',
    'fused_swiglu_ffn_fwd',
    'fused_two_mm_same_out_interface',
    'fused_two_mm_same_out_wT_xT_triton',
    'fused_two_mm_same_out_wT_x_triton',
    'fused_two_mm_swiglu_triton',
    'l2_norm_add_fused',
    'reference_l2_norm_add_fused',
    'reference_l2_norm_add_fused_with_momentum',
    'reference_lact_swiglu_ffn_fast_weight_grads',
    'reference_swiglu_ffn_fwd',
    'swiglu_backward_three_bmm_ref',
    'swiglu_backward_three_bmm_triton',
    'swiglu_backward_three_bmm_with_lr_triton',
    'triton_swiglu_bwd_bwd_fused_cat_inp_out',
]
