
import math
import pathlib
import re

import pytest
import torch

from fla.models import LaCTConfig, LaCTForCausalLM
from fla.utils import device

from .test_modeling_base import run_test_model_forward_backward

# LaCT only updates its fast weights once per `lact_chunk_size` tokens, so the default 2048 would
# leave the test-time-training path completely dormant at test sequence lengths. Shrink it so the
# update loop is actually exercised.
LACT_KWARGS = dict(lact_chunk_size=64, window_size=128)


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-{}".format(*test))
        for test in [
            (2, 2, 512, 4, 64, False, torch.float16),
            (2, 2, 512, 4, 64, True, torch.float16),
        ]
    ],
)
def test_modeling(L: int, B: int, T: int, H: int, D: int, use_l2warp: bool, dtype: torch.dtype):
    run_test_model_forward_backward(L, B, T, H, D, LaCTConfig, use_l2warp, dtype, **LACT_KWARGS)


# ===================================================================================
# Explicit forward/backward coverage
# ===================================================================================
# `run_test_model_forward_backward` only reaches its backward pass on the variable-length branch,
# which LaCT opts out of. Without this the opt-out would silently cost us all backward coverage.
@pytest.mark.parametrize('window_size', [128, None], ids=['swa', 'full-causal'])
def test_forward_backward(window_size):
    """A full training step must produce a sane loss and finite gradients everywhere."""
    B, T = 2, 512
    config = LaCTConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_heads=4,
        num_lact_heads=2,
        lact_chunk_size=64,
        window_size=window_size,
        vocab_size=1000,
        fuse_cross_entropy=False,
    )
    model = LaCTForCausalLM(config).to(device).to(torch.float16)
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)

    out = model(input_ids=input_ids, labels=input_ids)
    assert out.logits.shape == (B, T, config.vocab_size)
    assert torch.isfinite(out.loss), "loss is not finite"
    # a fresh model should sit near the uniform-prior loss; wildly off means the forward is broken
    assert abs(out.loss.item() - math.log(config.vocab_size)) < 0.5, \
        f"loss {out.loss.item():.3f} is far from ln(vocab_size)={math.log(config.vocab_size):.3f}"

    out.loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"

    # the fast weights are the point of the layer; make sure they are actually being trained
    fast_weights = [n for n, _ in model.named_parameters() if n.endswith(('.w0', '.w1', '.w2'))]
    assert len(fast_weights) == 3 * config.num_hidden_layers, \
        f"expected 3 fast weights per layer, found {fast_weights}"


def test_rejects_cu_seqlens():
    """Packed input must fail loudly rather than leak fast-weight state across documents."""
    config = LaCTConfig(hidden_size=128, num_hidden_layers=1, num_heads=4, num_lact_heads=2,
                        lact_chunk_size=64, window_size=128, vocab_size=500)
    model = LaCTForCausalLM(config).to(device).to(torch.float16)
    input_ids = torch.randint(0, 500, (1, 256), device=device)
    cu_seqlens = torch.tensor([0, 128, 256], dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError, match='cu_seqlens'):
        model(input_ids=input_ids, cu_seqlens=cu_seqlens)


def test_no_flash_attn_dependency():
    """The whole point of the port: nothing on the LaCT path may import `flash_attn`."""
    import fla.layers.lact
    import fla.models.lact.modeling_lact
    import fla.ops.lact

    # match real import statements only -- prose mentioning the package is fine
    pattern = re.compile(r'^\s*(?:from\s+flash_attn|import\s+flash_attn)', re.MULTILINE)
    roots = [pathlib.Path(fla.ops.lact.__file__).parent,
             pathlib.Path(fla.layers.lact.__file__),
             pathlib.Path(fla.models.lact.modeling_lact.__file__).parent]
    checked = 0
    for root in roots:
        files = sorted(root.rglob('*.py')) if root.is_dir() else [root]
        for path in files:
            source = path.read_text(encoding='utf-8')
            assert not pattern.search(source), f"{path} imports flash_attn"
            checked += 1
    assert checked > 10, f"expected to scan the whole LaCT tree, only saw {checked} files"


# NOTE: no generation tests. Upstream LaCT stores no fast-weight state in the cache, so incremental
# decoding is only correct while the sequence stays within one chunk. `LaCTConfig` is listed in
# GENERATION_UNSUPPORTED for that reason; see the port notes.
