from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from transformers_compat import patch_optional_sklearn

patch_optional_sklearn()
from transformers import AutoModel, BertModel


TEXT_FEATURE_DIM = 768


class TextFeatureAdapter(nn.Module):
    """Residual MLP adapter for aligning text features before PWAM fusion."""

    def __init__(self, dim: int = TEXT_FEATURE_DIM, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x):
        return x + self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))


def build_text_feature_adapter(args) -> nn.Module:
    if getattr(args, "no_text_adapter", False):
        return nn.Identity()
    return TextFeatureAdapter(dim=TEXT_FEATURE_DIM, dropout=getattr(args, "text_adapter_dropout", 0.1))


def build_text_encoder_model(args) -> nn.Module:
    """Build the language encoder while preserving RemoteSAM's 768-d text contract."""

    encoder_name = getattr(args, "text_encoder", "modernbert")
    if encoder_name == "bert":
        model_path = args.ck_bert
        _validate_local_path(model_path, "BERT")
        model = BertModel.from_pretrained(model_path, local_files_only=True)
    elif encoder_name == "modernbert":
        model_path = args.modernbert_path
        _validate_local_path(model_path, "ModernBERT")
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
    else:
        raise ValueError(f"Unsupported text encoder: {encoder_name}")

    if hasattr(model, "pooler"):
        model.pooler = None

    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size != TEXT_FEATURE_DIM:
        raise ValueError(
            f"RemoteSAM PWAM expects {TEXT_FEATURE_DIM}-d text features, "
            f"but {encoder_name} provides hidden_size={hidden_size}."
        )

    # Some ModernBERT checkpoints are stored in fp16. RemoteSAM's PWAM/CIM and
    # decoder are trained in fp32 here, so keep the text encoder in fp32 too.
    return model.float()


def _validate_local_path(model_path: str, model_name: str) -> None:
    path = Path(model_path).expanduser()
    if path.is_absolute() or model_path.startswith("."):
        if not path.exists():
            raise FileNotFoundError(f"{model_name} model directory not found: {path}")


def _get_nested_attr(module: nn.Module, path: str):
    current = module
    for item in path.split("."):
        if not hasattr(current, item):
            return None
        current = getattr(current, item)
    return current


def _as_layer_list(layers) -> list[nn.Module]:
    if isinstance(layers, nn.ModuleList):
        return list(layers)
    if isinstance(layers, (list, tuple)) and all(isinstance(layer, nn.Module) for layer in layers):
        return list(layers)
    return []


def get_text_encoder_layers(text_encoder: nn.Module) -> list[nn.Module]:
    """Return transformer blocks for BERT-like and ModernBERT-like HuggingFace models."""

    for path in (
        "encoder.layer",
        "encoder.layers",
        "layers",
        "model.layers",
        "transformer.layer",
        "transformer.h",
    ):
        layers = _as_layer_list(_get_nested_attr(text_encoder, path))
        if layers:
            return layers

    raise AttributeError(
        f"Could not locate transformer layers in {text_encoder.__class__.__name__}; "
        "expected one of encoder.layer, encoder.layers, layers, model.layers, "
        "transformer.layer, or transformer.h."
    )


def get_text_encoder_trainable_params(text_encoder: nn.Module, max_layers: int = 10) -> list[nn.Parameter]:
    layers = get_text_encoder_layers(text_encoder)
    params: list[nn.Parameter] = []
    for layer in layers[: min(max_layers, len(layers))]:
        params.extend(parameter for parameter in layer.parameters() if parameter.requires_grad)

    for projection_name in ("output_projection", "projection"):
        projection = getattr(text_encoder, projection_name, None)
        if isinstance(projection, nn.Module):
            params.extend(parameter for parameter in projection.parameters() if parameter.requires_grad)

    if not params:
        raise ValueError(f"No trainable text parameters found in {text_encoder.__class__.__name__}.")
    return params
