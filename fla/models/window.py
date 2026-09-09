# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Per-layer sliding-window schedules, applied without per-module changes.

Every attention-family layer in FLA follows two conventions:

* it records the window it was built with as ``self.window_size`` and its
  position as ``self.layer_idx``;
* it reads ``self.window_size`` inside ``forward`` -- nothing derived from the
  window is precomputed in ``__init__``, and the KV cache is told about it per
  call via ``cache_kwargs=dict(window_size=self.window_size)``, which
  :class:`fla.models.utils.FLALayer` applies per layer.

So a window *schedule* can be applied after a model is constructed by assigning
``self.window_size`` layer by layer. That is behaviourally identical to having
built each layer with its own window, and it needs no change to any individual
layer or model -- including ones added to FLA later, as long as they keep those
two conventions.

Typical use::

    model = TransformerForCausalLM(config)
    apply_window_schedule(model, [512, 512, 1024, 2048, None, ...])

or, declaratively, by putting the schedule on the config and letting the
``post_init`` hook installed by :func:`install_window_schedule_hook` apply it.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    'WindowSchedule',
    'apply_window_schedule',
    'expand_window_schedule',
    'install_window_schedule_hook',
    'iter_windowed_modules',
]

#: A schedule is a single window (uniform, the pre-existing behaviour), a
#: sequence of windows positionally assigned to layers, or a mapping from layer
#: index to window that leaves unlisted layers untouched.
WindowSchedule = int | None | Sequence[int | None] | Mapping[int, int | None]

#: Attribute a module must expose to be treated as windowed.
_WINDOW_ATTR = 'window_size'


def _is_window_value(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value > 0)


def _check_window_value(value: object, *, where: str) -> int | None:
    if not _is_window_value(value):
        raise ValueError(f"{where} must be a positive integer or None; got {value!r}")
    return value


def _constructor_default_window(module: nn.Module) -> int | None:
    """The window ``module``'s own constructor would have used by default."""
    try:
        default = inspect.signature(type(module).__init__).parameters[_WINDOW_ATTR].default
    except (KeyError, TypeError, ValueError):
        return None
    if default is inspect.Parameter.empty:
        return None
    return default if _is_window_value(default) else None


def _accepts_none_window(module: nn.Module) -> bool:
    """Whether ``module``'s constructor declares ``window_size`` as optional.

    Read off the annotation rather than a hard-coded module list, so modules
    added later classify themselves. Layers written as ``window_size: int``
    (Rodimus' shared-key attention, which does ``max(self.window_size, q_len)``)
    are not given ``None``; the ``int | None`` layers are.
    """
    try:
        parameter = inspect.signature(type(module).__init__).parameters[_WINDOW_ATTR]
    except (KeyError, TypeError, ValueError):
        # No introspectable signature: assume the permissive case and let the
        # layer itself complain if it cannot cope.
        return True
    annotation = parameter.annotation
    if annotation is inspect.Parameter.empty:
        return parameter.default is None
    return 'None' in str(annotation)


def iter_windowed_modules(model: nn.Module) -> Iterator[tuple[int, str, nn.Module]]:
    """Yield ``(layer_idx, qualified_name, module)`` for every windowed module.

    A module qualifies when it carries both a ``window_size`` attribute and an
    integer ``layer_idx``. Results are ordered by layer index, then by name, so
    a model with several windowed modules in one layer stays deterministic.
    """
    found: list[tuple[int, str, nn.Module]] = []
    for name, module in model.named_modules():
        if not hasattr(module, _WINDOW_ATTR):
            continue
        layer_idx = getattr(module, 'layer_idx', None)
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int) or layer_idx < 0:
            continue
        found.append((layer_idx, name, module))
    found.sort(key=lambda item: (item[0], item[1]))
    yield from found


def expand_window_schedule(
    schedule: WindowSchedule,
    *,
    num_hidden_layers: int,
    windowed_layers: Sequence[int] | None = None,
) -> dict[int, int | None]:
    """Normalize ``schedule`` into a ``{layer_idx: window_size}`` mapping.

    Accepted forms:

    ``None`` or ``int``
        One window for every layer -- the pre-existing behaviour.
    sequence of length ``num_hidden_layers``
        One entry per model layer, positionally.
    sequence of length ``len(windowed_layers)``
        One entry per *windowed* layer, positionally. This is the useful form
        for hybrid models, where only a few layers run standard attention: the
        i-th entry belongs to the i-th attention layer, in layer order.
    mapping
        Sparse ``{layer_idx: window_size}``. Layers absent from the mapping keep
        whatever window they were constructed with.

    ``windowed_layers`` is the sorted list of layer indices that actually have a
    window; pass the indices discovered on a built model so the per-attention
    -layer form can be resolved. Without it, only the whole-model length is
    accepted.
    """
    if isinstance(num_hidden_layers, bool) or not isinstance(num_hidden_layers, int) or num_hidden_layers < 0:
        raise ValueError(f"'num_hidden_layers' must be a non-negative integer; got {num_hidden_layers!r}")

    targets = list(range(num_hidden_layers)) if windowed_layers is None else list(windowed_layers)

    if schedule is None or isinstance(schedule, int) and not isinstance(schedule, bool):
        _check_window_value(schedule, where="'window_size'")
        return dict.fromkeys(targets, schedule)

    if isinstance(schedule, Mapping):
        expanded: dict[int, int | None] = {}
        for key, value in schedule.items():
            layer_idx = int(key)  # JSON round-trips integer keys as strings
            if layer_idx < 0 or layer_idx >= num_hidden_layers:
                raise ValueError(
                    f"'window_size' maps out-of-range layer {layer_idx!r}; "
                    f"expected a value in [0, {num_hidden_layers})",
                )
            expanded[layer_idx] = _check_window_value(value, where=f"'window_size' entry for layer {layer_idx}")
        return expanded

    if isinstance(schedule, Sequence) and not isinstance(schedule, (str, bytes)):
        values = list(schedule)
        if len(values) == num_hidden_layers:
            keys = list(range(num_hidden_layers))
        elif windowed_layers is not None and len(values) == len(targets):
            keys = targets
        else:
            expected = f"{num_hidden_layers}"
            if windowed_layers is not None and len(targets) != num_hidden_layers:
                expected = f"{num_hidden_layers} (one per layer) or {len(targets)} (one per windowed layer)"
            raise ValueError(
                f"'window_size' has {len(values)} entries but the model expects {expected}",
            )
        return {
            layer_idx: _check_window_value(value, where=f"'window_size' entry for layer {layer_idx}")
            for layer_idx, value in zip(keys, values, strict=True)
        }

    raise ValueError(
        "'window_size' must be an integer, None, a sequence of them, or a "
        f"{{layer_idx: window_size}} mapping; got {schedule!r}",
    )


def apply_window_schedule(
    model: nn.Module,
    schedule: WindowSchedule,
    *,
    num_hidden_layers: int | None = None,
) -> dict[str, int | None]:
    """Assign a per-layer window schedule to an already-constructed model.

    Returns ``{module_name: window_size}`` for the modules that were changed,
    which makes the call easy to assert on in tests.

    Raises if the schedule would hand ``None`` to a layer whose constructor
    declares ``window_size: int`` (i.e. one that has no full-attention mode).
    """
    windowed = list(iter_windowed_modules(model))
    if not windowed:
        if schedule is None:
            return {}
        raise ValueError(
            'a window schedule was given but the model has no modules exposing '
            "both 'window_size' and 'layer_idx'",
        )

    windowed_layers = sorted({layer_idx for layer_idx, _, _ in windowed})
    if num_hidden_layers is None:
        num_hidden_layers = getattr(getattr(model, 'config', None), 'num_hidden_layers', None)
        if num_hidden_layers is None:
            num_hidden_layers = windowed_layers[-1] + 1

    expanded = expand_window_schedule(
        schedule,
        num_hidden_layers=num_hidden_layers,
        windowed_layers=windowed_layers,
    )

    if isinstance(schedule, Mapping):
        # A mapping names its layers explicitly, so a key that matches no
        # windowed layer is a mistake worth reporting rather than ignoring.
        # (A full-length sequence is not checked: its entries for non-attention
        # layers are expected to be inert.)
        unmatched = sorted(set(expanded) - set(windowed_layers))
        if unmatched:
            raise ValueError(
                f"'window_size' assigns layer(s) {unmatched} that have no windowed module; "
                f'the windowed layers are {windowed_layers}',
            )

    applied: dict[str, int | None] = {}
    for layer_idx, name, module in windowed:
        if layer_idx in expanded:
            window_size = expanded[layer_idx]
        elif _is_window_value(getattr(module, _WINDOW_ATTR, None)):
            # A sparse mapping leaves this layer alone -- it already holds a
            # usable window.
            continue
        else:
            # The block forwarded the raw schedule (a list/mapping) into the
            # layer, so there is no usable window there to keep. Fall back to
            # what the layer's own constructor would have defaulted to.
            window_size = _constructor_default_window(module)
        if window_size is None and not _accepts_none_window(module):
            raise ValueError(
                f"layer {layer_idx} ({name}, {type(module).__name__}) requires a finite window; "
                'it declares `window_size: int` and has no full-attention mode, so None cannot be scheduled for it',
            )
        module.window_size = window_size
        applied[name] = window_size
    return applied


def install_window_schedule_hook(*classes: type, attribute: str = 'window_size') -> None:
    """Make ``classes`` apply ``config.<attribute>`` as a schedule after building.

    Wraps ``post_init`` -- which every FLA model calls at the end of
    ``__init__``, including on the ``from_pretrained`` path -- so a schedule
    stored on the config is applied automatically and survives
    ``save_pretrained``/``from_pretrained``. Re-applying is idempotent, so the
    double call made by ``XForCausalLM`` (once for the inner model, once for
    itself) is harmless.
    """
    for cls in classes:
        if getattr(cls, '_fla_window_schedule_hooked', False):
            continue
        original_post_init = cls.post_init

        def post_init(self, *args: Any, _original=original_post_init, **kwargs: Any) -> None:
            _original(self, *args, **kwargs)
            schedule = getattr(self.config, attribute, None)
            # A plain int/None is what the blocks already applied at build time;
            # only a sequence or mapping needs a second pass.
            if isinstance(schedule, (Sequence, Mapping)) and not isinstance(schedule, (str, bytes)):
                apply_window_schedule(self, schedule, num_hidden_layers=self.config.num_hidden_layers)

        cls.post_init = post_init
        cls._fla_window_schedule_hooked = True
