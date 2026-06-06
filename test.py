from __future__ import annotations

import numpy as np
import torch
import torch.utils.data

import transforms as T
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


def checkpoint_has_modal_lora(checkpoint) -> bool:
    state_dict = checkpoint.get("model", checkpoint)
    return any(is_modal_lora_parameter(key) for key in state_dict.keys())


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


def get_transform(args):
    return T.Compose(
        [
            T.Resize(args.img_size, args.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def compute_iou(prediction: np.ndarray, target: np.ndarray):
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return intersection, union


def modal_to_tensor(modal, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(modal):
        return modal.to(device=device, dtype=torch.long).reshape(-1)
    if isinstance(modal, str):
        return torch.tensor([1 if "sar" in modal.lower() else 0], device=device, dtype=torch.long)
    if isinstance(modal, (list, tuple)):
        values = []
        for item in modal:
            if torch.is_tensor(item):
                values.extend(int(value) for value in item.reshape(-1).tolist())
            elif isinstance(item, str):
                values.append(1 if "sar" in item.lower() else 0)
            else:
                values.append(int(item))
        return torch.tensor(values, device=device, dtype=torch.long)
    return torch.as_tensor(modal, device=device, dtype=torch.long).reshape(-1)


@torch.no_grad()
def evaluate(model, data_loader, bert_model, device):
    model.eval()
    if bert_model is not None:
        bert_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    thresholds = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    precision_hits = np.zeros(len(thresholds), dtype=np.int64)
    mean_ious = []
    mean_ious_rgb = []
    mean_ious_sar = []
    cum_i, cum_u = 0, 0
    cum_i_rgb, cum_u_rgb = 0, 0
    cum_i_sar, cum_u_sar = 0, 0

    for image, target, sentences, attentions, _, modal in metric_logger.log_every(data_loader, 100, "Test:"):
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

            modal_id = int(modal[index].item()) if index < modal.numel() else 0
            if modal_id == 1:
                mean_ious_sar.append(this_iou)
                cum_i_sar += intersection
                cum_u_sar += union
            else:
                mean_ious_rgb.append(this_iou)
                cum_i_rgb += intersection
                cum_u_rgb += union

    mean_iou = float(np.mean(mean_ious)) if mean_ious else 0.0
    mean_iou_rgb = float(np.mean(mean_ious_rgb)) if mean_ious_rgb else 0.0
    mean_iou_sar = float(np.mean(mean_ious_sar)) if mean_ious_sar else 0.0
    overall_iou = 100.0 * cum_i / cum_u if cum_u > 0 else 0.0
    overall_iou_rgb = 100.0 * cum_i_rgb / cum_u_rgb if cum_u_rgb > 0 else 0.0
    overall_iou_sar = 100.0 * cum_i_sar / cum_u_sar if cum_u_sar > 0 else 0.0
    print("Final results:")
    print("Mean IoU is %.2f" % (mean_iou * 100.0))
    print("    Mean IoU RGB = %.2f (%d samples)" % (mean_iou_rgb * 100.0, len(mean_ious_rgb)))
    print("    Mean IoU SAR = %.2f (%d samples)" % (mean_iou_sar * 100.0, len(mean_ious_sar)))
    for threshold, hits in zip(thresholds, precision_hits):
        print("    precision@%s = %.2f" % (threshold, hits * 100.0 / max(1, len(mean_ious))))
    print("    overall IoU = %.2f" % overall_iou)
    print("    overall IoU RGB = %.2f" % overall_iou_rgb)
    print("    overall IoU SAR = %.2f" % overall_iou_sar)


def load_weights(model, checkpoint):
    state_dict = checkpoint.get("model", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {remove_prefix(key, "module."): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)


def maybe_set_window_size_from_checkpoint(args, checkpoint) -> None:
    state_dict = checkpoint.get("model", checkpoint)
    for key, value in state_dict.items():
        key = remove_prefix(key, "module.")
        if key.endswith("relative_position_bias_table") and value.ndim == 2:
            table_size = int(value.shape[0] ** 0.5)
            window_size = (table_size + 1) // 2
            if window_size == 12 and not args.window12:
                print("Detected window size 12 from checkpoint; enabling --window12 for evaluation.")
                args.window12 = True
            return


def main(args):
    if not args.resume:
        raise ValueError("--resume is required for testing")

    checkpoint = torch_load_checkpoint(args.resume, map_location="cpu")
    if checkpoint_has_modal_lora(checkpoint) and not args.use_modal_lora:
        args.use_modal_lora = True
        print("Detected modal LoRA weights in checkpoint; enabling --use_modal_lora.")
    maybe_set_window_size_from_checkpoint(args, checkpoint)

    device = torch.device(args.device)
    dataset_test = All_Dataset(
        args,
        split=args.split,
        max_tokens=args.max_tokens,
        image_transforms=get_transform(args),
        eval_mode=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=torch.utils.data.SequentialSampler(dataset_test),
        num_workers=args.workers,
        pin_memory=args.pin_mem,
    )

    model = segmentation.__dict__[args.model](pretrained="", args=args)
    load_weights(model, checkpoint)
    model = model.to(device)

    if args.model == "lavt_one":
        bert_model = None
    else:
        bert_model = build_text_encoder_model(args)
        if "bert_model" in checkpoint:
            bert_state = checkpoint["bert_model"]
            if any(key.startswith("module.") for key in bert_state):
                bert_state = {remove_prefix(key, "module."): value for key, value in bert_state.items()}
            bert_model.load_state_dict(bert_state, strict=False)
        bert_model = bert_model.to(device)

    evaluate(model, data_loader_test, bert_model, device)


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    print(f"Image size: {args.img_size}")
    main(args)
