"""Select recovered datasets without modifying the original dataset factory."""
from data.DiverseDataset import DiverseDataset as original_dataset
from tasks.code._adapters.datasets_mlc import MSAR_MLC
from tasks.code._adapters.datasets_vg import SARDet_VG


def DiverseDataset(args, image_transforms=None, max_tokens=20):
    if args.task == "MLC" and args.dataset == "MSAR":
        return MSAR_MLC(args, max_tokens=max_tokens)
    if args.task == "VG" and args.dataset == "SARDet":
        return SARDet_VG(args, max_tokens=max_tokens)
    return original_dataset(args, image_transforms=image_transforms, max_tokens=max_tokens)
