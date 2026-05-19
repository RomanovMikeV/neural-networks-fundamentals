import torch
from torch import nn
from typing import Iterable, Tuple, List

@torch.no_grad()
def deterministic_init(
    model: nn.Module,
    start: float = -1.0,
    end:   float =  1.0,
    include_bias: bool = True,
    only_trainable: bool = True,
) -> None:
    """
    Fill all floating-point parameters of `model` with values from a single
    global linspace(start, end, steps=total_num_elements) in a *stable* order.

    Why this is deterministic:
      1) We collect parameters by *name* and sort them lexicographically.
         That breaks any dependency on module construction/registration order.
      2) We use a single contiguous linspace and assign slices in the sorted order.

    Args:
        model: torch.nn.Module to initialize (in-place).
        start, end: endpoints for the global linspace.
        include_bias: include parameters whose name ends with ".bias".
        only_trainable: if True, only params with requires_grad=True are filled.

    Notes:
        - We only touch floating-point tensors (skip ints/bools).
        - Works on CPU/GPU models; the master linspace is built on CPU and sliced
          onto each parameter's device/dtype to avoid large GPU allocations.
        - Result depends on the *names* present. Changing architecture or
          renaming modules will change assignments (as intended).
    """
    # (name, parameter) list, filtered and sorted for stability
    params: List[Tuple[str, torch.Tensor]] = []
    for name, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        if only_trainable and not p.requires_grad:
            continue
        if (not include_bias) and name.endswith(".bias"):
            continue
        params.append((name, p))

    # Stable order: sort by full parameter name
    params.sort(key=lambda kv: kv[0])

    # Count total number of elements
    total = sum(p.numel() for _, p in params)
    if total == 0:
        return

    # Build a single global vector on CPU for reproducibility and low GPU mem use
    master = torch.linspace(start, end, steps=total, dtype=torch.float32, device="cpu")

    # Assign contiguous slices to each parameter, reshaped to its tensor shape
    idx = 0
    for _, p in params:
        n = p.numel()
        slice_ = master[idx:idx+n].to(device=p.device, dtype=p.dtype, non_blocking=False)
        p.copy_(slice_.view_as(p))
        idx += n