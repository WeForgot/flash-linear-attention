# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import json
import os

import pytest
import torch

import fla.layers.attn as attn_module
import fla.layers.mla as mla_module
from fla.layers.attn import Attention
from fla.layers.rodimus import SlidingWindowSharedKeyAttention
from fla.models import GLAConfig, GLAForCausalLM, MLAConfig, MLAForCausalLM, TransformerConfig, TransformerForCausalLM
from fla.models.window import (
    _accepts_none_window,
    apply_window_schedule,
    expand_window_schedule,
    iter_windowed_modules,
)

# The schedule is applied at construction time, so these tests only need the
# modules to be constructible; they never run an attention kernel.
pytestmark = pytest.mark.skipif(
    attn_module.flash_attn_func is None or mla_module.flash_attn_func is None,
    reason='flash-attn is required to construct the attention layers',
)

BASE = dict(hidden_size=64, num_heads=4, num_kv_heads=4, vocab_size=128)


def windows(model):
    return [getattr(layer.attn, 'window_size', ...) for layer in model.model.layers]


# ---------------------------------------------------------------- expansion --

def test_expand_scalar_and_none_are_uniform():
    assert expand_window_schedule(None, num_hidden_layers=3) == {0: None, 1: None, 2: None}
    assert expand_window_schedule(512, num_hidden_layers=3) == {0: 512, 1: 512, 2: 512}


def test_expand_sequence_per_layer():
    assert expand_window_schedule([1, 2, None], num_hidden_layers=3) == {0: 1, 1: 2, 2: None}


def test_expand_sequence_per_windowed_layer():
    # one entry per *attention* layer, the useful form for hybrid models
    assert expand_window_schedule(
        [256, 1024], num_hidden_layers=8, windowed_layers=[1, 4],
    ) == {1: 256, 4: 1024}


def test_expand_mapping_is_sparse_and_accepts_string_keys():
    assert expand_window_schedule({1: 256}, num_hidden_layers=4) == {1: 256}
    # config.json round-trips integer keys as strings
    assert expand_window_schedule({'1': 256}, num_hidden_layers=4) == {1: 256}


@pytest.mark.parametrize(
    ('schedule', 'match'),
    [
        ([1, 2], 'expects 3'),
        ([0, 1, 2], 'positive integer or None'),
        ([1, 2, -4], 'positive integer or None'),
        ({7: 128}, 'out-of-range layer'),
        ('512', 'must be an integer'),
    ],
)
def test_expand_rejects_bad_schedules(schedule, match):
    with pytest.raises(ValueError, match=match):
        expand_window_schedule(schedule, num_hidden_layers=3)


# ------------------------------------------------------------- non-hybrid ---

def test_scalar_window_is_unchanged():
    model = TransformerForCausalLM(TransformerConfig(num_hidden_layers=3, window_size=512, **BASE))
    assert windows(model) == [512, 512, 512]


def test_sequence_window_is_applied_per_layer():
    model = TransformerForCausalLM(
        TransformerConfig(num_hidden_layers=4, window_size=[128, 128, 256, None], **BASE),
    )
    assert windows(model) == [128, 128, 256, None]


def test_mapping_window_leaves_other_layers_at_their_default():
    model = TransformerForCausalLM(TransformerConfig(num_hidden_layers=4, window_size={1: 256, 3: 1024}, **BASE))
    assert windows(model) == [None, 256, None, 1024]


def test_schedule_works_for_a_second_model_family():
    model = MLAForCausalLM(
        MLAConfig(hidden_size=64, num_heads=4, vocab_size=128, num_hidden_layers=3, window_size=[64, 128, None]),
    )
    assert windows(model) == [64, 128, None]


def test_scheduled_layer_matches_one_built_with_that_window():
    """The whole approach rests on this: assigning after construction is the
    same as having constructed the layer with that window."""
    scheduled = TransformerForCausalLM(TransformerConfig(num_hidden_layers=2, window_size=[128, 256], **BASE))
    for layer_idx, window_size in enumerate((128, 256)):
        reference = TransformerForCausalLM(
            TransformerConfig(num_hidden_layers=2, window_size=window_size, **BASE),
        )

        def plain(module):
            return {
                k: v for k, v in module.__dict__.items()
                if not k.startswith('_') and not isinstance(v, (torch.Tensor, torch.nn.Module))
            }

        assert plain(scheduled.model.layers[layer_idx].attn) == plain(reference.model.layers[layer_idx].attn)
    assert [n for n, _ in scheduled.named_parameters()] == [
        n for n, _ in TransformerForCausalLM(TransformerConfig(num_hidden_layers=2, window_size=128, **BASE))
        .named_parameters()
    ]


def test_schedule_survives_save_and_load(tmp_path):
    schedule = [128, 256, 512, None]
    model = TransformerForCausalLM(TransformerConfig(num_hidden_layers=4, window_size=schedule, **BASE))
    model.save_pretrained(tmp_path)

    with open(os.path.join(tmp_path, 'config.json')) as f:
        assert json.load(f)['window_size'] == schedule

    assert windows(TransformerForCausalLM.from_pretrained(tmp_path)) == schedule


# ----------------------------------------------------------------- hybrid ---

def test_hybrid_list_of_specs_gives_per_layer_windows():
    """Hybrid models need no new code at all: one spec per window."""
    model = GLAForCausalLM(GLAConfig(
        hidden_size=64, num_heads=4, vocab_size=128, num_hidden_layers=8,
        attn=[
            {'layers': [1], 'num_heads': 4, 'window_size': 512},
            {'layers': [4], 'num_heads': 4, 'window_size': 2048},
            {'layers': [7], 'num_heads': 4},
        ],
    ))
    assert [getattr(layer.attn, 'window_size', ...) for layer in model.model.layers] == \
        [..., 512, ..., ..., 2048, ..., ..., None]


def test_hybrid_accepts_one_entry_per_attention_layer():
    model = GLAForCausalLM(GLAConfig(
        hidden_size=64, num_heads=4, vocab_size=128, num_hidden_layers=8,
        attn={'layers': [1, 4, 7], 'num_heads': 4, 'window_size': 512},
    ))
    applied = apply_window_schedule(model, [256, 1024, None])
    assert applied == {
        'model.layers.1.attn': 256,
        'model.layers.4.attn': 1024,
        'model.layers.7.attn': None,
    }
    assert [getattr(layer.attn, 'window_size', ...) for layer in model.model.layers] == \
        [..., 256, ..., ..., 1024, ..., ..., None]


def test_hybrid_rejects_a_wrong_length_schedule():
    model = GLAForCausalLM(GLAConfig(
        hidden_size=64, num_heads=4, vocab_size=128, num_hidden_layers=8,
        attn={'layers': [1, 4, 7], 'num_heads': 4, 'window_size': 512},
    ))
    with pytest.raises(ValueError, match='8 .one per layer. or 3 .one per windowed layer.'):
        apply_window_schedule(model, [256, 1024])


# ---------------------------------------------------------------- discovery --

def test_iter_windowed_modules_finds_attention_layers_in_order():
    model = TransformerForCausalLM(TransformerConfig(num_hidden_layers=3, window_size=256, **BASE))
    assert [(i, name) for i, name, _ in iter_windowed_modules(model)] == [
        (0, 'model.layers.0.attn'),
        (1, 'model.layers.1.attn'),
        (2, 'model.layers.2.attn'),
    ]


# -------------------------------------------------------------------- guard --

def test_layers_without_a_full_attention_mode_never_receive_none():
    assert _accepts_none_window(Attention(hidden_size=64, num_heads=4, window_size=None, layer_idx=0)) is True
    swa = SlidingWindowSharedKeyAttention(hidden_size=64, num_heads=4, window_size=2048, layer_idx=0)
    assert _accepts_none_window(swa) is False

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = swa

    with pytest.raises(ValueError, match='requires a finite window'):
        apply_window_schedule(Holder(), {0: None}, num_hidden_layers=1)


def test_mapping_naming_a_non_attention_layer_is_reported():
    model = GLAForCausalLM(GLAConfig(
        hidden_size=64, num_heads=4, vocab_size=128, num_hidden_layers=8,
        attn={'layers': [1, 4], 'num_heads': 4, 'window_size': 512},
    ))
    with pytest.raises(ValueError, match=r'layer\(s\) \[2\] that have no windowed module'):
        apply_window_schedule(model, {1: 256, 2: 512})
