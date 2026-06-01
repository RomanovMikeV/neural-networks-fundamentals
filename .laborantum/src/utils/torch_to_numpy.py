from collections.abc import Mapping

import torch


def torch_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, Mapping):
        return type(value)(
            (key, torch_to_numpy(item))
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(torch_to_numpy(item) for item in value)
    if isinstance(value, list):
        return [torch_to_numpy(item) for item in value]
    return value
