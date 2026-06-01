import math
from typing import Set

import torch
from torch import nn


def _should_touch(parameter: torch.Tensor, only_trainable: bool) -> bool:
    return (
        parameter.is_floating_point()
        and (not only_trainable or parameter.requires_grad)
    )


@torch.no_grad()
def deterministic_init(
    model: nn.Module,
    start: float = -1.0,
    end: float = 1.0,
    include_bias: bool = True,
    only_trainable: bool = True,
) -> None:
    """
    Initialize layer weights with a deterministic linspace scaled by fan-in.

    For each module weight tensor with at least two dimensions, values are set to
    linspace(start, end, steps=weight.numel()) / sqrt(fan_in), where fan_in is
    the number of input signals for one output unit.

    Biases are set to zero when include_bias=True. Other floating parameters are
    filled with an unscaled linspace as a deterministic fallback.
    """
    initialized: Set[int] = set()

    for module in model.modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
            if _should_touch(weight, only_trainable):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                scale = math.sqrt(fan_in) if fan_in > 0 else 1.0
                values = torch.linspace(
                    start,
                    end,
                    steps=weight.numel(),
                    dtype=torch.float32,
                    device="cpu",
                )
                values = values.to(device=weight.device, dtype=weight.dtype)
                weight.copy_(values.view_as(weight) / scale)
                initialized.add(id(weight))

        bias = getattr(module, "bias", None)
        if include_bias and isinstance(bias, torch.Tensor):
            if _should_touch(bias, only_trainable):
                bias.zero_()
                initialized.add(id(bias))

    for _, parameter in model.named_parameters():
        if id(parameter) in initialized:
            continue
        if not _should_touch(parameter, only_trainable):
            continue
        if parameter.ndim == 0:
            parameter.fill_(float(start))
            continue
        values = torch.linspace(
            start,
            end,
            steps=parameter.numel(),
            dtype=torch.float32,
            device="cpu",
        )
        values = values.to(device=parameter.device, dtype=parameter.dtype)
        parameter.copy_(values.view_as(parameter))
