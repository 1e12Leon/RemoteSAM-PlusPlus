"""Compatibility helpers for importing Transformers in lean training envs."""

from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec


def patch_optional_sklearn() -> None:
    """Avoid broken optional sklearn imports inside newer Transformers.

    Some Transformers versions import generation helpers while importing BERT.
    Those helpers optionally import ``sklearn.metrics.roc_curve``. The training
    code never uses text generation, so a local sklearn/numpy ABI mismatch should
    not prevent BERT from loading.
    """

    try:
        from sklearn.metrics import roc_curve  # noqa: F401
        return
    except Exception:
        pass

    sklearn_module = types.ModuleType("sklearn")
    metrics_module = types.ModuleType("sklearn.metrics")
    sklearn_module.__spec__ = ModuleSpec("sklearn", loader=None)
    metrics_module.__spec__ = ModuleSpec("sklearn.metrics", loader=None)

    def roc_curve(*args, **kwargs):
        raise ImportError("sklearn is unavailable in this environment")

    metrics_module.roc_curve = roc_curve
    sklearn_module.metrics = metrics_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.metrics"] = metrics_module
