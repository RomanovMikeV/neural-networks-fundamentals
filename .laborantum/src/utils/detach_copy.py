from __future__ import annotations
from dataclasses import is_dataclass, fields
from collections import abc
from typing import Any
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def detach_copy(obj: Any) -> Any:
    """
    Recursively copy `obj`, turning torch.Tensor into NumPy arrays detached from the graph.
    - torch.Tensor -> tensor.detach().cpu().numpy().copy()
    - NumPy arrays -> .copy()
    - Mappings, sequences, sets, namedtuples, dataclasses -> same type with recursively processed elements
    - Scalars/strings/None -> returned as-is
    """
    # --- Torch tensors ---
    if _HAS_TORCH and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().copy()

    # --- Trivial immutables ---
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj

    # --- NumPy arrays ---
    if isinstance(obj, np.ndarray):
        return obj.copy()

    # --- Dataclasses ---
    if is_dataclass(obj) and not isinstance(obj, type):
        kwargs = {}
        for f in fields(obj):
            kwargs[f.name] = detach_copy(getattr(obj, f.name))
        return type(obj)(**kwargs)

    # --- Namedtuple ---
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(detach_copy(x) for x in obj))

    # --- Mappings (dict and similar) ---
    if isinstance(obj, abc.Mapping):
        try:
            return type(obj)((detach_copy(k), detach_copy(v)) for k, v in obj.items())
        except Exception:
            return {detach_copy(k): detach_copy(v) for k, v in obj.items()}

    # --- Sequences ---
    if isinstance(obj, list):
        return [detach_copy(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(detach_copy(x) for x in obj)

    # --- Sets ---
    if isinstance(obj, set):
        return {detach_copy(x) for x in obj}
    if isinstance(obj, frozenset):
        return frozenset(detach_copy(x) for x in obj)

    # --- Fallback: return as-is ---
    return obj
