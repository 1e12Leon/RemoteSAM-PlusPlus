from __future__ import annotations

import copy

import numpy as np
import torch
import torch.utils.data

import utils
from data.all_dataset import All_Dataset
from lib import segmentation
from lib.text_encoders import build_text_encoder_model
from test import (
    checkpoint_has_modal_lora,
    compute_iou,
    get_transform,
    load_weights,
    maybe_set_window_size_from_checkpoint,
    modal_to_tensor,
    remove_prefix,
    torch_load_checkpoint,
)


MODALITIES = {
    "rgb": {
        "id": 0,
        "name": "RGB",
        "image_arg": "rgb_imageFolder",
        "ref_arg": "rgb_ref_file_path",
        "instances_arg": "rgb_instances_path",
        "flag": "test_rgb",
    },
    "sar": {
        "id": 1,
        "name": "SAR",
        "image_arg": "sar_imageFolder",
        "ref_arg": "sar_ref_file_path",
        "instances_arg": "sar_instances_path",
        "flag": "test_sar",
    },
}


class ForcedModalDataset(torch.utils.data.Dataset):
    """Wrap an existing dataset and force the modal id used by modal LoRA."""

    def __init__(self, dataset: torch.utils.data.Dataset, modal_id: int) -> None:
        self.dataset = dataset
        self.modal_id = int(modal_id)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target, input_ids, attention_mask, exist, _ = self.dataset[index]
        modal = torch.tensor(self.modal_id, dtype=torch.long)
        return image, target, input_ids, attention_mask, exist, modal


def add_modal_test_args(parser):
    parser.add_argument("--test_rgb", action="store_true", help="evaluate the RGB test set")
    parser.add_argument("--test_sar", action="store_true", help="evaluate the SAR test set")
    parser.add_argument("--rgb_imageFolder", default="", help="RGB image folder")
    parser.add_argument("--rgb_ref_file_path", default="", help="RGB refs pickle path")
    parser.add_argument("--rgb_instances_path", default="", help="RGB instances.json path")
    parser.add_argument("--sar_imageFolder", default="", help="SAR image folder")
    parser.add_argument("--sar_ref_file_path", default="", help="SAR refs pickle path")
    parser.add_argument("--sar_instances_path", default="", help="SAR instances.json path")
    return parser


def selected_modalities(args) -> list[str]:
    selected = [key for key, cfg in MODALITIES.items() if getattr(args, cfg["flag"])]
    return selected if selected else list(MODALITIES.keys())


def require_modality_paths(args, modality: str) -> None:
    cfg = MODALITIES[modality]
    missing = [
        option
        for option, attr in (
            (f"--{cfg['image_arg']}", cfg["image_arg"]),
            (f"--{cfg['ref_arg']}", cfg["ref_arg"]),
            (f"--{cfg['instances_arg']}", cfg["instances_arg"]),
        )
        if not getattr(args, attr)
    ]
    if missing:
        raise ValueError(f"{cfg['name']} evaluation requires: {', '.join(missing)}")


def build_modality_dataset(args, modality: str):
    require_modality_paths(args, modality)
    cfg = MODALITIES[modality]
    modal_args = copy.copy(args)
    modal_args.imageFolder = getattr(args, cfg["image_arg"])
    modal_args.ref_file_path = getattr(args, cfg["ref_arg"])
    modal_args.instances_path = getattr(args, cfg["instances_arg"])

    dataset = All_Dataset(
        modal_args,
        split=args.split,
        max_tokens=args.max_tokens,
        image_transforms=get_transform(args),
        eval_mode=True,
    )
    return ForcedModalDataset(dataset, cfg["id"])


@torch.no_grad()
def evaluate_modality(model, data_loader, bert_model, device, modality_name: str):
    model.eval()
    if bert_model is not None:
        bert_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    thresholds = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    precision_hits = np.zeros(len(thresholds), dtype=np.int64)
    mean_ious = []
    cum_i, cum_u = 0, 0

    for image, target, sentences, attentions, _, modal in metric_logger.log_every(
        data_loader, 100, f"Test {modality_name}:"
    ):
        image = image.to(device, non_blocking=True)
        target_np = target.numpy()
        sentences = sentences.squeeze(1).to(device, non_blocking=True)
        attentions = attentions.squeeze(1).to(device, non_blocking=True)
        modal = modal_to_tensor(modal, image.device)

        if bert_model is None:
            output = model(image, sentences, attentions, modal=modal)
        else:
            hidden_states = bert_model(sentences, attention_mask=attentions)[0]
            embedding = hidden_states.permute(0, 2, 1).to(dtype=image.dtype)
            output = model(image, embedding, attentions.unsqueeze(-1), modal=modal)

        prediction = output.argmax(1).cpu().numpy()
        for index in range(prediction.shape[0]):
            intersection, union = compute_iou(prediction[index], target_np[index])
            this_iou = 0.0 if union == 0 else intersection / union
            mean_ious.append(this_iou)
            cum_i += intersection
            cum_u += union
            precision_hits += this_iou >= thresholds

    mean_iou = float(np.mean(mean_ious)) if mean_ious else 0.0
    overall_iou = 100.0 * cum_i / cum_u if cum_u > 0 else 0.0
    metrics = {
        "samples": len(mean_ious),
        "mIoU": mean_iou * 100.0,
        "oIoU": overall_iou,
        "precision": {
            str(threshold): hits * 100.0 / max(1, len(mean_ious))
            for threshold, hits in zip(thresholds, precision_hits)
        },
    }

    print(f"Final {modality_name} results:")
    print("    samples = %d" % metrics["samples"])
    print("    Mean IoU = %.2f" % metrics["mIoU"])
    print("    overall IoU = %.2f" % metrics["oIoU"])
    for threshold, precision in metrics["precision"].items():
        print("    precision@%s = %.2f" % (threshold, precision))
    return metrics


def build_model_and_text_encoder(args, checkpoint, device):
    model = segmentation.__dict__[args.model](pretrained="", args=args)
    load_weights(model, checkpoint)
    model = model.to(device)

    if args.model == "lavt_one":
        return model, None

    bert_model = build_text_encoder_model(args)
    if "bert_model" in checkpoint:
        bert_state = checkpoint["bert_model"]
        if any(key.startswith("module.") for key in bert_state):
            bert_state = {remove_prefix(key, "module."): value for key, value in bert_state.items()}
        bert_model.load_state_dict(bert_state, strict=False)
    return model, bert_model.to(device)


def main(args):
    if not args.resume:
        raise ValueError("--resume is required for testing")

    checkpoint = torch_load_checkpoint(args.resume, map_location="cpu")
    if checkpoint_has_modal_lora(checkpoint) and not args.use_modal_lora:
        args.use_modal_lora = True
        print("Detected modal LoRA weights in checkpoint; enabling --use_modal_lora.")
    maybe_set_window_size_from_checkpoint(args, checkpoint)

    device = torch.device(args.device)
    model, bert_model = build_model_and_text_encoder(args, checkpoint, device)

    results = {}
    for modality in selected_modalities(args):
        cfg = MODALITIES[modality]
        dataset = build_modality_dataset(args, modality)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            sampler=torch.utils.data.SequentialSampler(dataset),
            num_workers=args.workers,
            pin_memory=args.pin_mem,
        )
        results[modality] = evaluate_modality(model, data_loader, bert_model, device, cfg["name"])

    print("Summary:")
    for modality, metrics in results.items():
        print(
            "    %s: samples=%d, mIoU=%.2f, oIoU=%.2f"
            % (MODALITIES[modality]["name"], metrics["samples"], metrics["mIoU"], metrics["oIoU"])
        )


if __name__ == "__main__":
    from args import get_parser

    parser = add_modal_test_args(get_parser())
    args = parser.parse_args()
    print(f"Image size: {args.img_size}")
    main(args)
