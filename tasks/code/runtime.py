from __future__ import annotations

import torch

import utils
from data.all_dataset import All_Dataset
from lib import segmentation
from lib.modal_lora import is_modal_lora_parameter
from lib.text_encoders import build_text_encoder_model


def torch_load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


def checkpoint_has_modal_lora(checkpoint) -> bool:
    state_dict = checkpoint.get("model", checkpoint)
    return any(is_modal_lora_parameter(key) for key in state_dict.keys())


def maybe_set_window_size_from_checkpoint(args, checkpoint) -> None:
    state_dict = checkpoint.get("model", checkpoint)
    for key, value in state_dict.items():
        key = remove_prefix(key, "module.")
        if key.endswith("relative_position_bias_table") and value.ndim == 2:
            table_size = int(value.shape[0] ** 0.5)
            window_size = (table_size + 1) // 2
            if window_size == 12 and not getattr(args, "window12", False):
                print("Detected window size 12 from checkpoint; enabling --window12 for evaluation.")
                args.window12 = True
            return


def modal_to_id(modal: str | int | torch.Tensor | None) -> int:
    if modal is None:
        return 0
    if torch.is_tensor(modal):
        return int(modal.reshape(-1)[0].item())
    if isinstance(modal, int):
        return modal
    return 1 if "sar" in str(modal).lower() else 0


class ForcedModalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, modal_id: int) -> None:
        self.dataset = dataset
        self.modal_id = int(modal_id)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target, input_ids, attention_mask, exist, _ = self.dataset[index]
        modal = torch.tensor(self.modal_id, dtype=torch.long)
        return image, target, input_ids, attention_mask, exist, modal


def build_ref_dataset(args, split, transform):
    dataset = All_Dataset(
        args,
        split=split,
        max_tokens=args.max_tokens,
        image_transforms=transform,
        eval_mode=True,
    )
    return ForcedModalDataset(dataset, modal_to_id(getattr(args, "modal", "rgb")))


def load_model_and_text_encoder(args, *, distributed: bool = False):
    if not args.resume:
        raise ValueError("--resume is required")

    checkpoint = torch_load_checkpoint(args.resume, map_location="cpu")
    if checkpoint_has_modal_lora(checkpoint) and not getattr(args, "use_modal_lora", False):
        args.use_modal_lora = True
        print("Detected modal LoRA weights in checkpoint; enabling --use_modal_lora.")
    maybe_set_window_size_from_checkpoint(args, checkpoint)

    model = segmentation.__dict__[args.model](pretrained="", args=args)
    state_dict = checkpoint.get("model", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {remove_prefix(key, "module."): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    bert_model = None
    if args.model != "lavt_one":
        bert_model = build_text_encoder_model(args)
        if "bert_model" in checkpoint:
            bert_state = checkpoint["bert_model"]
            if any(key.startswith("module.") for key in bert_state):
                bert_state = {remove_prefix(key, "module."): value for key, value in bert_state.items()}
            bert_model.load_state_dict(bert_state, strict=False)

    if distributed:
        device = torch.device(f"cuda:{args.local_rank}")
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        if bert_model is not None:
            bert_model = bert_model.to(device)
            bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
    else:
        device = torch.device(args.device)
        model = model.to(device)
        if bert_model is not None:
            bert_model = bert_model.to(device)

    return model, bert_model, device


def gather_object(value):
    if not utils.is_dist_avail_and_initialized():
        return [value]
    gathered = [None for _ in range(utils.get_world_size())]
    torch.distributed.all_gather_object(gathered, value)
    return gathered
