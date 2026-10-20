import sys
import os
# set working directory to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import json
import torch
import torch.utils.data
import utils
import time
import numpy as np
from PIL import Image
from tasks.code._adapters.datasets_recovered import DiverseDataset
from tasks.code.model_auto import RemoteSAM
from tasks.code.runtime import gather_object, load_model_and_text_encoder
import warnings
warnings.filterwarnings("ignore")


def average_precision(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = y_true.sum()
    if positives <= 0:
        return np.nan
    order = np.argsort(-y_score)
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true)
    precision = tp / (np.arange(len(sorted_true), dtype=np.float64) + 1.0)
    return float((precision * sorted_true).sum() / positives)


def multilabel_ranking_metrics(predictions, classnames):
    dedup = {}
    for item in predictions:
        dedup.setdefault(item["image"], item)
    predictions = list(dedup.values())
    if not predictions:
        return {}

    y_true = np.zeros((len(predictions), len(classnames)), dtype=np.float64)
    y_score = np.zeros_like(y_true)
    class_to_idx = {classname: idx for idx, classname in enumerate(classnames)}
    for row_idx, item in enumerate(predictions):
        for classname in item.get("gt_classes") or []:
            if classname in class_to_idx:
                y_true[row_idx, class_to_idx[classname]] = 1.0
        scores = item.get("scores") or {}
        for classname, score in scores.items():
            if classname in class_to_idx:
                y_score[row_idx, class_to_idx[classname]] = float(score)

    y_pred = y_score > 0.5
    tp = np.logical_and(y_true == 1, y_pred).sum(axis=0).astype(np.float64)
    fp = np.logical_and(y_true == 0, y_pred).sum(axis=0).astype(np.float64)
    fn = np.logical_and(y_true == 1, ~y_pred).sum(axis=0).astype(np.float64)
    per_class_f1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1e-15)
    micro_tp, micro_fp, micro_fn = tp.sum(), fp.sum(), fn.sum()
    micro_f1 = 2.0 * micro_tp / max(2.0 * micro_tp + micro_fp + micro_fn, 1e-15)
    class_ap = np.array([average_precision(y_true[:, idx], y_score[:, idx]) for idx in range(len(classnames))])
    valid_ap = class_ap[~np.isnan(class_ap)]
    return {
        "sample_count": len(predictions),
        "mAP": float(valid_ap.mean()) if len(valid_ap) else float("nan"),
        "micro_f1": float(micro_f1),
        "macro_f1": float(per_class_f1.mean()),
        "class_ap": class_ap,
        "per_class_f1": per_class_f1,
    }


def eval(class_count, stats, predictions=None, classnames=None):
    tp, tn, fp, fn = stats.cpu().double()

    accuracy_total = (tp.sum() + tn.sum()) / (tp.sum() + tn.sum() + fp.sum() + fn.sum() + 1e-15)
    accuracy_per_class = (tp + tn) / (tp + tn + fp + fn + 1e-15)

    # sumarize evaluation
    print("\n########## Evaluation Summary ##########")
    print("Accuracy: {:.2f}%".format(accuracy_total.item() * 100.))
    print("Accuracy per class: {}".format(
        "  ".join([str(round(acc.item() * 100., 2)) + "%" for acc in accuracy_per_class])
    ))
    if predictions is not None and classnames is not None:
        ranking = multilabel_ranking_metrics(predictions, classnames)
        if ranking:
            print("Unique samples: {}".format(ranking["sample_count"]))
            print("mAP: {:.2f}%".format(ranking["mAP"] * 100.0))
            print("Micro-F1@0.5: {:.2f}%".format(ranking["micro_f1"] * 100.0))
            print("Macro-F1@0.5: {:.2f}%".format(ranking["macro_f1"] * 100.0))
            print("AP per class: {}".format(
                "  ".join(
                    "{}={:.2f}%".format(classname, ap * 100.0)
                    for classname, ap in zip(classnames, ranking["class_ap"])
                )
            ))
    print()


def classification(model, data_loader, bert_model, args, device):
    model = RemoteSAM(model, device, use_EPOC=args.EPOC, args=args, text_encoder=bert_model, EPOC_threshold = 0.25)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    classnames = data_loader.dataset.classes
    has_groundtruth = getattr(data_loader.dataset, "has_groundtruth", True)
    stats = torch.zeros((4, len(classnames)), dtype=torch.double, device=device)
    predictions = []
    local_total = 0
    start_time = time.time()

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, 'Test:'):
            origin_image, groundtruth, image_name = data

            groundtruth = groundtruth.squeeze(0).numpy()
            image_name = image_name[0]

            pred_ins, prob_ins = model.multi_label_cls(image=origin_image.squeeze(0).numpy(), classnames=classnames, return_prob=True)

            if has_groundtruth:
                true_tensor = torch.as_tensor(groundtruth, dtype=torch.bool, device=device)
                pred_tensor = torch.as_tensor([classname in pred_ins for classname in classnames], dtype=torch.bool, device=device)
                stats[0] += (true_tensor & pred_tensor).double()
                stats[1] += ((~true_tensor) & (~pred_tensor)).double()
                stats[2] += ((~true_tensor) & pred_tensor).double()
                stats[3] += (true_tensor & (~pred_tensor)).double()

            predictions.append({
                "image": image_name,
                "pred_classes": pred_ins,
                "scores": {classname: float(prob_ins[classname]) for classname in classnames},
                "gt_classes": [
                    classname for classname, exists in zip(classnames, groundtruth.tolist())
                    if exists
                ] if has_groundtruth else None,
            })
            local_total += 1

            del groundtruth, origin_image, image_name, pred_ins, prob_ins
            
            #! only for debug
            # if cum_total == 10:
            #     break

    if utils.is_dist_avail_and_initialized():
        print(
            "Rank {} finished MLC inference: {} samples, {:.1f} minutes. Waiting for metric sync.".format(
                utils.get_rank(),
                local_total,
                (time.time() - start_time) / 60.0,
            ),
            force=True,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

    gathered_predictions = gather_object(predictions)
    predictions = sum(gathered_predictions, [])
    predictions.sort(key=lambda item: item["image"])

    return classnames, stats, has_groundtruth, predictions


def save_predictions(args, predictions):
    os.makedirs(args.save_path, exist_ok=True)
    result_path = os.path.join(args.save_path, f"{args.dataset}_MLC_predictions.json")
    with open(result_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved MLC predictions to {result_path}")


def main(args):
    utils.init_distributed_mode(args)

    dataset_test = DiverseDataset(args)
    if utils.is_dist_avail_and_initialized():
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test,
            num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
        batch_size=1, sampler=test_sampler, num_workers=args.workers, pin_memory=args.pin_mem)

    model, bert_model, device = load_model_and_text_encoder(
        args, distributed=utils.is_dist_avail_and_initialized()
    )
    classnames, stats, has_groundtruth, predictions = classification(model, data_loader_test, bert_model, args=args, device=device)

    if utils.get_rank() == 0:
        save_predictions(args, predictions)
        if has_groundtruth:
            eval(len(classnames), stats, predictions=predictions, classnames=classnames)
        else:
            print("\n########## Evaluation Summary ##########")
            print("Ground-truth annotations were not found; skipped Accuracy computation.")
            print("Only prediction JSON was generated.")
            print()


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)
