from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _log(logger, message: str) -> None:
    if logger is not None:
        logger.warning(message)
    else:
        print(message)


def _unwrap_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a valid state dict.")

    keys = list(state_dict.keys())
    if keys and keys[0].startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    keys = list(state_dict.keys())
    if keys and keys[0].startswith("backbone."):
        print("Start stripping upper net pre-fix and loading backbone weights to our swin encoder")
        state_dict = {
            key.replace("backbone.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
    keys = sorted(state_dict.keys())
    if keys and keys[0].startswith("encoder"):
        state_dict = {
            key.replace("encoder.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
    return state_dict


def _resize_absolute_pos_embed(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    logger=None,
) -> None:
    if "absolute_pos_embed" not in state_dict or not hasattr(model, "absolute_pos_embed"):
        return
    absolute_pos_embed = state_dict["absolute_pos_embed"]
    current = model.absolute_pos_embed
    if absolute_pos_embed.ndim != 3 or current.ndim != 4:
        _log(logger, "Error in loading absolute_pos_embed, pass")
        return
    n_pretrained, length, channels_pretrained = absolute_pos_embed.size()
    n_current, channels_current, height, width = current.size()
    if n_pretrained != n_current or channels_pretrained != channels_current or length != height * width:
        _log(logger, "Error in loading absolute_pos_embed, pass")
        return
    state_dict["absolute_pos_embed"] = absolute_pos_embed.view(
        n_current,
        height,
        width,
        channels_current,
    ).permute(0, 3, 1, 2)


def _resize_relative_position_bias(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    logger=None,
) -> None:
    model_state = model.state_dict()
    table_keys = [key for key in state_dict if "relative_position_bias_table" in key]
    for table_key in table_keys:
        if table_key not in model_state:
            continue
        table_pretrained = state_dict[table_key]
        table_current = model_state[table_key]
        length_pretrained, heads_pretrained = table_pretrained.size()
        length_current, heads_current = table_current.size()
        if heads_pretrained != heads_current:
            _log(logger, f"Error in loading {table_key}, pass")
            continue
        if length_pretrained == length_current:
            continue
        size_pretrained = int(length_pretrained**0.5)
        size_current = int(length_current**0.5)
        table_resized = F.interpolate(
            table_pretrained.permute(1, 0).view(1, heads_pretrained, size_pretrained, size_pretrained),
            size=(size_current, size_current),
            mode="bicubic",
        )
        state_dict[table_key] = table_resized.view(heads_current, length_current).permute(1, 0)


def load_checkpoint(
    model: nn.Module,
    filename: str,
    map_location: str = "cpu",
    strict: bool = False,
    logger=None,
) -> dict[str, Any]:
    checkpoint_path = Path(filename).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{checkpoint_path} is not a checkpoint file")

    checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {checkpoint_path}")

    state_dict = OrderedDict(_unwrap_state_dict(checkpoint))
    _resize_absolute_pos_embed(model, state_dict, logger)
    _resize_relative_position_bias(model, state_dict, logger)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if strict and (missing_keys or unexpected_keys):
        message = (
            "The model and loaded state dict do not match exactly\n"
            f"missing keys: {missing_keys}\n"
            f"unexpected keys: {unexpected_keys}"
        )
        raise RuntimeError(message)
    return checkpoint
