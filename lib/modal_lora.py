from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalHardGatedLoRALinear(nn.Module):
    """Linear layer with separate RGB/SAR LoRA branches.

    The base linear projection is kept structurally identical to the original
    module. A per-sample modal gate selects the RGB or SAR low-rank residual.
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.05,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}.")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.weight.requires_grad = linear.weight.requires_grad and not freeze_base
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
            self.bias.requires_grad = linear.bias.requires_grad and not freeze_base
        else:
            self.register_parameter("bias", None)

        self.lora_A_rgb = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B_rgb = nn.Parameter(torch.empty(self.out_features, r))
        self.lora_A_sar = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B_sar = nn.Parameter(torch.empty(self.out_features, r))
        self.modal: torch.Tensor | None = None
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        for lora_a in (self.lora_A_rgb, self.lora_A_sar):
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        for lora_b in (self.lora_B_rgb, self.lora_B_sar):
            nn.init.zeros_(lora_b)

    def set_modal(self, modal) -> None:
        if modal is None:
            self.modal = None
            return
        modal_tensor = torch.as_tensor(modal, dtype=torch.long)
        self.modal = modal_tensor.reshape(-1)

    def _expanded_modal(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.modal is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        modal = self.modal.to(device=device, dtype=torch.long)
        if modal.numel() == batch_size:
            return modal
        if batch_size % modal.numel() == 0:
            return modal.repeat_interleave(batch_size // modal.numel())
        if modal.numel() == 1:
            return modal.expand(batch_size)
        raise ValueError(
            f"Modal batch size mismatch: got {modal.numel()} modal labels for linear input batch {batch_size}."
        )

    def _lora(self, x: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(self.dropout(x), lora_a), lora_b) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        rgb_delta = self._lora(x, self.lora_A_rgb, self.lora_B_rgb)
        sar_delta = self._lora(x, self.lora_A_sar, self.lora_B_sar)

        modal = self._expanded_modal(x.shape[0], x.device)
        sar_mask = (modal == 1).to(dtype=out.dtype)
        while sar_mask.ndim < out.ndim:
            sar_mask = sar_mask.unsqueeze(-1)
        return out + rgb_delta * (1.0 - sar_mask) + sar_delta * sar_mask


def normalize_modal_lora_targets(target_names: Iterable[str] | str | None) -> tuple[str, ...]:
    if target_names is None:
        return tuple()
    if isinstance(target_names, str):
        target_names = [item.strip() for item in target_names.split(",")]
    return tuple(item for item in target_names if item)


def replace_linear_with_modal_lora(
    module: nn.Module,
    *,
    r: int,
    alpha: float,
    dropout: float,
    target_names: Iterable[str] | str,
    freeze_base: bool = False,
) -> int:
    """Replace selected child Linear modules with modal-gated LoRA wrappers."""

    targets = normalize_modal_lora_targets(target_names)
    if not targets:
        return 0

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(name == target or name.endswith(target) for target in targets):
            setattr(
                module,
                name,
                ModalHardGatedLoRALinear(
                    child,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    freeze_base=freeze_base,
                ),
            )
            replaced += 1
        else:
            replaced += replace_linear_with_modal_lora(
                child,
                r=r,
                alpha=alpha,
                dropout=dropout,
                target_names=targets,
                freeze_base=freeze_base,
            )
    return replaced


def set_modal_for_lora(module: nn.Module, modal) -> None:
    for child in module.modules():
        if isinstance(child, ModalHardGatedLoRALinear):
            child.set_modal(modal)


def is_modal_lora_parameter(name: str) -> bool:
    return ".lora_" in name or name.startswith("lora_")

