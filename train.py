# RemoteSAM++ distributed referring-segmentation training.
from __future__ import annotations

import datetime
import os
import random
import time

import numpy as np
import torch
import torch.utils.data

import transforms as T
import utils
from data.all_dataset import All_Dataset
from lib import segmentation
from lib.modal_lora import is_modal_lora_parameter
from lib.text_encoders import build_text_encoder_model, get_text_encoder_trainable_params
from loss.loss import Loss


def torch_load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_has_modal_lora(checkpoint) -> bool:
    state_dict = checkpoint.get("model", checkpoint)
    return any(is_modal_lora_parameter(key) for key in state_dict.keys())


try:
    import swanlab
except Exception:
    swanlab = None


def swanlab_enabled(args) -> bool:
    return bool(args.use_swanlab and swanlab is not None and utils.is_main_process())


def init_swanlab(args) -> None:
    if not args.use_swanlab:
        return
    if swanlab is None:
        if utils.is_main_process():
            print("swanlab is not installed; continue training without swanlab logging.")
        return
    if not utils.is_main_process():
        return

    config = vars(args).copy()
    init_kwargs = {
        "project": args.swanlab_project,
        "config": config,
    }
    if args.swanlab_run_name:
        init_kwargs["name"] = args.swanlab_run_name
    if args.swanlab_mode:
        init_kwargs["mode"] = args.swanlab_mode
    swanlab.init(**init_kwargs)


def log_swanlab(args, metrics: dict, step: int | None = None) -> None:
    if not swanlab_enabled(args):
        return
    if step is None:
        swanlab.log(metrics)
    else:
        swanlab.log(metrics, step=step)


def seed_everything(seed: int = 2401) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset(image_set: str, transform, args, eval_mode: bool):
    dataset = All_Dataset(
        args,
        split=image_set,
        max_tokens=args.max_tokens,
        image_transforms=transform,
        eval_mode=eval_mode,
    )
    return dataset, 2


def get_transform(args):
    return T.Compose(
        [
            T.Resize(args.img_size, args.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def criterion(input, target, weight: float = 0.1):
    return Loss(weight=weight, device=input.device)(input, target)


def compute_iou(output: torch.Tensor, target: torch.Tensor):
    pred = output.argmax(1).bool()
    gt = target.bool()
    intersection = torch.logical_and(pred, gt).sum()
    union = torch.logical_or(pred, gt).sum()
    iou = torch.zeros((), device=output.device) if union == 0 else intersection.float() / union.float()
    return iou.item(), intersection.item(), union.item()


def compute_iou_per_sample(output: torch.Tensor, target: torch.Tensor):
    pred = output.argmax(1).bool()
    gt = target.bool()
    reduce_dims = tuple(range(1, pred.ndim))
    intersection = torch.logical_and(pred, gt).sum(dim=reduce_dims)
    union = torch.logical_or(pred, gt).sum(dim=reduce_dims)
    iou = torch.where(union > 0, intersection.float() / union.float(), torch.zeros_like(union, dtype=torch.float32))
    return iou.detach().cpu().tolist(), intersection.detach().cpu().tolist(), union.detach().cpu().tolist()


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


def encode_text(bert_model, sentences, attentions, dtype=None):
    last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
    embedding = last_hidden_states.permute(0, 2, 1)
    if dtype is not None:
        embedding = embedding.to(dtype=dtype)
    return embedding, attentions.unsqueeze(dim=-1)


@torch.no_grad()
def evaluate(model, data_loader, bert_model, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    total_loss = 0.0
    total_steps = 0
    cum_i, cum_u = 0, 0
    cum_i_rgb, cum_u_rgb = 0, 0
    cum_i_sar, cum_u_sar = 0, 0
    mean_ious = []
    mean_ious_rgb = []
    mean_ious_sar = []
    thresholds = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    precision_hits = np.zeros(len(thresholds), dtype=np.int64)

    for image, target, sentences, attentions, _, modal in metric_logger.log_every(data_loader, 100, header):
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        sentences = sentences.squeeze(1).cuda(non_blocking=True)
        attentions = attentions.squeeze(1).cuda(non_blocking=True)
        modal = modal_to_tensor(modal, image.device)

        if bert_model is None:
            output = model(image, sentences, attentions, modal=modal)
        else:
            embedding, language_mask = encode_text(bert_model, sentences, attentions, dtype=image.dtype)
            output = model(image, embedding, language_mask, modal=modal)

        loss = criterion(output, target)
        ious, intersections, unions = compute_iou_per_sample(output, target)

        total_loss += loss.item()
        total_steps += 1
        cum_i += int(sum(intersections))
        cum_u += int(sum(unions))
        mean_ious.extend(ious)
        precision_hits += (np.asarray(ious)[:, None] >= thresholds[None, :]).sum(axis=0)

        for index, (iou, intersection, union) in enumerate(zip(ious, intersections, unions)):
            modal_id = int(modal[index].item()) if index < modal.numel() else 0
            if modal_id == 1:
                cum_i_sar += int(intersection)
                cum_u_sar += int(union)
                mean_ious_sar.append(iou)
            else:
                cum_i_rgb += int(intersection)
                cum_u_rgb += int(union)
                mean_ious_rgb.append(iou)

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

    metrics = {
        "mIoU": mean_iou * 100.0,
        "mIoU_rgb": mean_iou_rgb * 100.0,
        "mIoU_sar": mean_iou_sar * 100.0,
        "oIoU": overall_iou,
        "oIoU_rgb": overall_iou_rgb,
        "oIoU_sar": overall_iou_sar,
        "loss": total_loss / max(1, total_steps),
    }
    return metrics["mIoU"], metrics["oIoU"], metrics["loss"], metrics


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, epoch, print_freq, iterations, bert_model, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    train_loss = 0.0
    total_steps = 0

    for image, target, sentences, attentions, _, modal in metric_logger.log_every(data_loader, print_freq, header):
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        sentences = sentences.squeeze(1).cuda(non_blocking=True)
        attentions = attentions.squeeze(1).cuda(non_blocking=True)
        modal = modal_to_tensor(modal, image.device)

        if bert_model is None:
            output = model(image, sentences, attentions, modal=modal)
        else:
            embedding, language_mask = encode_text(bert_model, sentences, attentions, dtype=image.dtype)
            output = model(image, embedding, language_mask, modal=modal)

        loss = criterion(output, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        total_steps += 1
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        log_swanlab(
            args,
            {
                "train/loss_iter": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
            },
            step=iterations,
        )

    avg_loss = train_loss / max(1, total_steps)
    log_swanlab(args, {"train/loss_epoch": avg_loss, "train/epoch": epoch}, step=iterations)
    return iterations, avg_loss


def build_model(args):
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    if utils.is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True,
        )
    return model


def build_text_encoder(args):
    if args.model == "lavt_one":
        return None

    bert_model = build_text_encoder_model(args)
    bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model).cuda()
    if utils.is_dist_avail_and_initialized():
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
    return bert_model


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def is_no_decay_parameter(name: str) -> bool:
    return "norm" in name or "absolute_pos_embed" in name or "relative_position_bias_table" in name


def is_adapter_no_decay_parameter(name: str) -> bool:
    return name.endswith("bias") or "norm" in name


def is_remote_sam_backbone_parameter(name: str) -> bool:
    remote_prefixes = (
        "fusions.",
        "res_gates.",
        "visual_residuals.",
        "visual_gates.",
        "norms.",
        "cim.",
    )
    remote_markers = (
        ".fusion.",
        ".res_gate.",
        ".visual_residual.",
        ".visual_gate.",
    )
    return name.startswith(remote_prefixes) or any(marker in name for marker in remote_markers)


def append_param_group(param_groups, params, *, lr: float, weight_decay: float, name: str) -> None:
    params = [parameter for parameter in params if parameter.requires_grad]
    if not params:
        return
    param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay, "name": name})


def build_optimizer_params(single_model, single_bert_model, args):
    backbone_lr = args.backbone_lr
    fusion_lr = args.fusion_lr if args.fusion_lr is not None else args.lr
    classifier_lr = args.classifier_lr if args.classifier_lr is not None else args.lr
    text_adapter_lr = args.text_adapter_lr if args.text_adapter_lr is not None else fusion_lr

    pretrained_decay, pretrained_no_decay = [], []
    modal_lora_params = []
    fusion_decay, fusion_no_decay = [], []
    for name, parameter in single_model.backbone.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_modal_lora_parameter(name):
            modal_lora_params.append(parameter)
            continue
        target = fusion_no_decay if is_remote_sam_backbone_parameter(name) and is_no_decay_parameter(name) else None
        if target is None and is_remote_sam_backbone_parameter(name):
            target = fusion_decay
        elif target is None and is_no_decay_parameter(name):
            target = pretrained_no_decay
        elif target is None:
            target = pretrained_decay
        target.append(parameter)

    text_encoder = single_model.text_encoder if single_bert_model is None else single_bert_model
    text_params = get_text_encoder_trainable_params(text_encoder, max_layers=10)
    text_adapter_decay, text_adapter_no_decay = [], []
    text_adapter = getattr(single_model, "text_adapter", None)
    if text_adapter is not None:
        for name, parameter in text_adapter.named_parameters():
            if is_adapter_no_decay_parameter(name):
                text_adapter_no_decay.append(parameter)
            else:
                text_adapter_decay.append(parameter)

    param_groups = []
    append_param_group(
        param_groups,
        modal_lora_params,
        lr=args.modal_lora_lr,
        weight_decay=0.0,
        name="modal_lora",
    )
    append_param_group(param_groups, pretrained_no_decay, lr=backbone_lr, weight_decay=0.0, name="visual_backbone_no_decay")
    append_param_group(param_groups, pretrained_decay, lr=backbone_lr, weight_decay=args.weight_decay, name="visual_backbone")
    append_param_group(param_groups, fusion_no_decay, lr=fusion_lr, weight_decay=0.0, name="remote_sam_fusion_no_decay")
    append_param_group(param_groups, fusion_decay, lr=fusion_lr, weight_decay=args.weight_decay, name="remote_sam_fusion")
    append_param_group(
        param_groups,
        single_model.classifier.parameters(),
        lr=classifier_lr,
        weight_decay=args.weight_decay,
        name="decoder",
    )
    append_param_group(
        param_groups,
        text_adapter_no_decay,
        lr=text_adapter_lr,
        weight_decay=0.0,
        name="text_adapter_no_decay",
    )
    append_param_group(
        param_groups,
        text_adapter_decay,
        lr=text_adapter_lr,
        weight_decay=args.weight_decay,
        name="text_adapter",
    )
    append_param_group(param_groups, text_params, lr=args.text_lr, weight_decay=args.weight_decay, name="text_encoder")
    if utils.is_main_process():
        print("Optimizer parameter groups:")
        for group in param_groups:
            num_params = sum(parameter.numel() for parameter in group["params"])
            print(
                f"  {group['name']}: lr={group['lr']}, "
                f"weight_decay={group['weight_decay']}, params={num_params}"
            )
    return param_groups


def main(args):
    dataset, _ = get_dataset("train", get_transform(args), args, eval_mode=False)
    dataset_test, _ = get_dataset("test", get_transform(args), args, eval_mode=True)

    if utils.is_dist_avail_and_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=utils.get_world_size(),
            rank=utils.get_rank(),
            shuffle=True,
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
    )

    checkpoint = None
    if args.resume:
        checkpoint = torch_load_checkpoint(args.resume, map_location="cpu")
        if checkpoint_has_modal_lora(checkpoint) and not args.use_modal_lora:
            args.use_modal_lora = True
            print("Detected modal LoRA weights in checkpoint; enabling --use_modal_lora.")

    model = build_model(args)
    bert_model = build_text_encoder(args)
    single_model = unwrap(model)
    single_bert_model = unwrap(bert_model) if bert_model is not None else None

    if checkpoint is not None:
        missing_keys, unexpected_keys = single_model.load_state_dict(checkpoint["model"], strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        if single_bert_model is not None and "bert_model" in checkpoint:
            single_bert_model.load_state_dict(checkpoint["bert_model"], strict=False)

    params_to_optimize = build_optimizer_params(single_model, single_bert_model, args)

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        amsgrad=args.amsgrad,
    )
    total_iterations = max(1, len(data_loader) * args.epochs)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (1 - step / total_iterations) ** 0.9,
    )

    resume_epoch = -1
    if checkpoint is not None and not args.eval:
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as exc:
                print(f"Skip optimizer resume because parameter groups changed: {exc}")
        if "lr_scheduler" in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            except ValueError as exc:
                print(f"Skip lr_scheduler resume because parameter groups changed: {exc}")
        resume_epoch = checkpoint.get("epoch", -1)

    if args.eval:
        evaluate(model, data_loader_test, bert_model, args)
        return

    start_time = time.time()
    iterations = 0
    best_miou = checkpoint.get("best_miou", -0.1) if checkpoint is not None else -0.1
    best_oiou = checkpoint.get("best_oiou", -0.1) if checkpoint is not None else -0.1
    for epoch in range(resume_epoch + 1, args.epochs):
        if hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(epoch)
        iterations, train_loss = train_one_epoch(
            model, optimizer, data_loader, lr_scheduler, epoch, args.print_freq, iterations, bert_model, args
        )
        mean_iou, overall_iou, val_loss, val_metrics = evaluate(model, data_loader_test, bert_model, args)
        print(f"Average object IoU {mean_iou}")
        print(f"Overall IoU {overall_iou}")

        is_best_miou = mean_iou > best_miou
        is_best_oiou = overall_iou > best_oiou
        if is_best_miou:
            best_miou = mean_iou
        if is_best_oiou:
            best_oiou = overall_iou

        checkpoint_to_save = {
            "model": single_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_miou": best_miou,
            "best_oiou": best_oiou,
            "mean_iou": mean_iou,
            "overall_iou": overall_iou,
        }
        if single_bert_model is not None:
            checkpoint_to_save["bert_model"] = single_bert_model.state_dict()

        if is_best_miou:
            print(f"Better mIoU epoch: {epoch}\n")
            utils.save_on_master(
                checkpoint_to_save,
                os.path.join(args.output_dir, f"model_best_mIoU_{args.model_id}.pth"),
            )
        if is_best_oiou:
            print(f"Better oIoU epoch: {epoch}\n")
            utils.save_on_master(
                checkpoint_to_save,
                os.path.join(args.output_dir, f"model_best_oIoU_{args.model_id}.pth"),
            )
            utils.save_on_master(
                checkpoint_to_save,
                os.path.join(args.output_dir, f"model_best_{args.model_id}.pth"),
            )
        utils.save_on_master(checkpoint_to_save, os.path.join(args.output_dir, f"model_last_{args.model_id}.pth"))
        log_swanlab(
            args,
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/mIoU": mean_iou,
                "val/oIoU": overall_iou,
                "val/mIoU_rgb": val_metrics["mIoU_rgb"],
                "val/mIoU_sar": val_metrics["mIoU_sar"],
                "val/oIoU_rgb": val_metrics["oIoU_rgb"],
                "val/oIoU_sar": val_metrics["oIoU_sar"],
                "val/loss": val_loss,
                "val/best_mIoU": best_miou,
                "val/best_oIoU": best_oiou,
            },
            step=iterations,
        )

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time}")


if __name__ == "__main__":
    from args import get_parser

    seed_everything()
    parser = get_parser()
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    init_swanlab(args)
    print(f"Image size: {args.img_size}")
    main(args)
