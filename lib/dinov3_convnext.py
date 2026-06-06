from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import PWAM
from .cross_scale_interaction import CIM
from .modal_lora import replace_linear_with_modal_lora, set_modal_for_lora
from .various_receptive import VariousReceptive

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:  # pragma: no cover
    load_safetensors = None


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class ConvNextLayerNorm(nn.LayerNorm):
    pass


class DINOv3ConvNextBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layer_norm = ConvNextLayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)
        self.pointwise_conv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = F.gelu(x)
        x = self.pointwise_conv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + x


class DINOv3ConvNextStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, stage_index: int):
        super().__init__()
        if stage_index == 0:
            self.downsample_layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4),
                LayerNorm2d(out_channels),
            )
        else:
            self.downsample_layers = nn.Sequential(
                LayerNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            )
        self.layers = nn.ModuleList([DINOv3ConvNextBlock(out_channels) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_layers(x)
        for layer in self.layers:
            x = layer(x)
        return x


class DINOv3ConvNextBackbone(nn.Module):
    """DINOv3 ConvNeXt-B wrapper that preserves RemoteSAM's 4-level contract."""

    def __init__(
        self,
        model_dir: str,
        num_heads_fusion: list[int] | None = None,
        fusion_drop: float = 0.0,
        use_modal_lora: bool = False,
        modal_lora_r: int = 4,
        modal_lora_alpha: float = 8.0,
        modal_lora_dropout: float = 0.05,
        modal_lora_stages: tuple[int, ...] = (2, 3),
        modal_lora_targets: tuple[str, ...] = ("pointwise_conv1", "pointwise_conv2"),
        modal_lora_freeze_base: bool = False,
    ):
        super().__init__()
        model_path = Path(model_dir)
        config = json.loads((model_path / "config.json").read_text())
        hidden_sizes = config["hidden_sizes"]
        depths = config["depths"]
        if num_heads_fusion is None:
            num_heads_fusion = [1, 1, 1, 1]
        if len(num_heads_fusion) != len(hidden_sizes):
            raise ValueError(f"num_heads_fusion must have {len(hidden_sizes)} values, got {num_heads_fusion}")

        self.num_features = hidden_sizes
        self.stages = nn.ModuleList()
        in_channels = config.get("num_channels", 3)
        for i, (out_channels, depth) in enumerate(zip(hidden_sizes, depths)):
            self.stages.append(DINOv3ConvNextStage(in_channels, out_channels, depth, i))
            in_channels = out_channels

        self.fusions = nn.ModuleList([
            PWAM(
                dim,
                dim,
                768,
                dim,
                dim,
                num_heads=num_heads_fusion[i],
                dropout=fusion_drop,
            )
            for i, dim in enumerate(hidden_sizes)
        ])
        self.res_gates = nn.ModuleList([self._make_gate(dim) for dim in hidden_sizes])
        self.visual_residuals = nn.ModuleList([VariousReceptive(dim) for dim in hidden_sizes])
        self.visual_gates = nn.ModuleList([self._make_gate(dim) for dim in hidden_sizes])
        self.norms = nn.ModuleList([ConvNextLayerNorm(dim, eps=config.get("layer_norm_eps", 1e-6)) for dim in hidden_sizes])
        self.layer_norm = ConvNextLayerNorm(hidden_sizes[-1], eps=config.get("layer_norm_eps", 1e-6))
        self.cim = CIM(dim=sum(hidden_sizes))
        self._init_remote_sam_weights()
        self.init_weights(model_path / "model.safetensors")
        if use_modal_lora:
            self.enable_modal_lora(
                r=modal_lora_r,
                alpha=modal_lora_alpha,
                dropout=modal_lora_dropout,
                stages=modal_lora_stages,
                targets=modal_lora_targets,
                freeze_base=modal_lora_freeze_base,
            )

    @staticmethod
    def _make_gate(dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh(),
        )

    def _init_remote_sam_weights(self) -> None:
        def init_module(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        for module in (
            self.fusions,
            self.res_gates,
            self.visual_residuals,
            self.visual_gates,
            self.norms,
            self.cim,
        ):
            module.apply(init_module)

    def init_weights(self, weights_path: Path) -> None:
        if load_safetensors is None:
            raise ImportError("safetensors is required to load DINOv3 ConvNeXt weights")
        state_dict = load_safetensors(str(weights_path), device="cpu")
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded DINOv3 ConvNeXt-B from {weights_path}")
        remote_sam_prefixes = ("fusions.", "res_gates.", "visual_residuals.", "visual_gates.", "norms.", "cim.")
        missing_pretrained = [key for key in missing if not key.startswith(remote_sam_prefixes)]
        missing_remote_sam = len(missing) - len(missing_pretrained)
        if missing_pretrained:
            print("DINOv3 missing pretrained keys:", missing_pretrained[:20])
        if missing_remote_sam:
            print(f"Initialized {missing_remote_sam} RemoteSAM fusion/CIM keys from scratch.")
        if unexpected:
            print("DINOv3 unexpected keys:", unexpected[:20])

    def enable_modal_lora(
        self,
        *,
        r: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.05,
        stages: tuple[int, ...] = (2, 3),
        targets: tuple[str, ...] = ("pointwise_conv1", "pointwise_conv2"),
        freeze_base: bool = False,
    ) -> None:
        replaced = 0
        for stage_index in stages:
            if stage_index < 0 or stage_index >= len(self.stages):
                raise ValueError(f"Invalid DINOv3 ConvNeXt stage index {stage_index}; expected 0-{len(self.stages) - 1}.")
            replaced += replace_linear_with_modal_lora(
                self.stages[stage_index],
                r=r,
                alpha=alpha,
                dropout=dropout,
                target_names=targets,
                freeze_base=freeze_base,
            )
        print(
            "Enabled modal LoRA for DINOv3 ConvNeXt-B: "
            f"targets={targets}, stages={stages}, rank={r}, alpha={alpha}, dropout={dropout}, "
            f"replaced={replaced}, freeze_base={freeze_base}"
        )

    def set_modal(self, modal) -> None:
        set_modal_for_lora(self, modal)

    def forward(self, x: torch.Tensor, l: torch.Tensor | None = None, l_mask: torch.Tensor | None = None):
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            bsz, channels, height, width = x.shape
            x_tokens = x.flatten(2).transpose(1, 2)

            if l is not None and l_mask is not None:
                v_residual = self.visual_residuals[i](x_tokens)
                x_residual = self.fusions[i](x_tokens, l, l_mask)
                x_tokens = (
                    x_tokens
                    + self.res_gates[i](x_residual) * x_residual
                    + self.visual_gates[i](v_residual) * v_residual
                )
                out_tokens = x_residual
            else:
                out_tokens = x_tokens

            out_tokens = self.norms[i](out_tokens)
            outs.append(out_tokens.reshape(bsz, height, width, channels).permute(0, 3, 1, 2).contiguous())
            x = x_tokens.transpose(1, 2).reshape(bsz, channels, height, width).contiguous()
        outs = self.cim(outs)
        return tuple(outs)
