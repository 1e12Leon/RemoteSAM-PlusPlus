from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


SAR_MODAL_ID = 1
RGB_MODAL_ID = 0
SAR_CHANNEL_DIFF_THRESHOLD = 1.0


def infer_modal_id_from_image(image: Any, threshold: float = SAR_CHANNEL_DIFF_THRESHOLD) -> int:
    """Infer RGB/SAR from image content, not dataset metadata.

    Rule:
    - 2D images are SAR-like.
    - RGB-like channel images with average channel difference <= threshold are SAR-like.
    - Otherwise they are RGB-like.
    """

    if torch.is_tensor(image):
        array = image.detach().cpu().numpy()
    elif isinstance(image, Image.Image):
        array = np.asarray(image)
    else:
        array = np.asarray(image)

    if array.ndim == 2:
        return SAR_MODAL_ID

    if array.ndim == 3:
        if array.shape[-1] >= 3:
            rgb = array[..., :3].astype(np.float32)
        elif array.shape[0] >= 3:
            rgb = np.moveaxis(array[:3], 0, -1).astype(np.float32)
        else:
            return SAR_MODAL_ID

        rg = np.abs(rgb[..., 0] - rgb[..., 1]).mean()
        rb = np.abs(rgb[..., 0] - rgb[..., 2]).mean()
        gb = np.abs(rgb[..., 1] - rgb[..., 2]).mean()
        avg_channel_diff = float((rg + rb + gb) / 3.0)
        return SAR_MODAL_ID if avg_channel_diff <= threshold else RGB_MODAL_ID

    return RGB_MODAL_ID


def infer_modal_tensor_from_image(
    image: Any,
    *,
    device: torch.device | str | None = None,
    threshold: float = SAR_CHANNEL_DIFF_THRESHOLD,
) -> torch.Tensor:
    return torch.tensor([infer_modal_id_from_image(image, threshold)], device=device, dtype=torch.long)
