from collections.abc import Mapping, Sequence
import torch

def collate_tensors_only(batch):
    """
    Collate rule:
      - torch.Tensor  -> stack on dim 0 (if shapes mismatch -> keep as list)
      - Mapping (dict)-> recurse per key; non-tensor leaves become List[..]
      - Sequence      -> elementwise recurse if all same length; else keep as List[..]
      - everything else (ints, floats, str, objects, numpy arrays, ...)-> List[..]
    """
    elem = batch[0]

    # 1) Tensors: try to stack
    if isinstance(elem, torch.Tensor):
        try:
            return torch.stack(batch, dim=0)
        except RuntimeError:
            # different shapes -> don't collate
            return list(batch)

    # 2) Dicts / mappings: union of keys, recurse per key
    if isinstance(elem, Mapping):
        keys = set().union(*(b.keys() for b in batch))
        return {k: collate_tensors_only([b.get(k) for b in batch]) for k in keys}

    # 3) Sequences (lists/tuples), but not strings/bytes
    if isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
        lens = [len(b) for b in batch]
        if len(set(lens)) != 1:
            return list(batch)  # different lengths -> keep raw
        transposed = list(zip(*batch))  # elementwise across the batch
        out = [collate_tensors_only(list(items)) for items in transposed]
        return type(elem)(out)  # preserve list/tuple type

    # 4) Everything else -> just keep as a list
    return list(batch)
