import sys
import os
# set working directory to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import json
import torch
import torch.utils.data
import utils
import time
from PIL import Image
from data.DiverseDataset import DiverseDataset
from tasks.code.model import RemoteSAM
from tasks.code.runtime import gather_object, load_model_and_text_encoder
import warnings
warnings.filterwarnings("ignore")


def eval(class_count, stats):
    tp, tn, fp, fn = stats.cpu().double()

    accuracy_total = (tp.sum() + tn.sum()) / (tp.sum() + tn.sum() + fp.sum() + fn.sum() + 1e-15)
    accuracy_per_class = (tp + tn) / (tp + tn + fp + fn + 1e-15)

    # sumarize evaluation
    print("\n########## Evaluation Summary ##########")
    print("Accuracy: {:.2f}%".format(accuracy_total.item() * 100.))
    print("Accuracy per class: {}".format(
        "  ".join([str(round(acc.item() * 100., 2)) + "%" for acc in accuracy_per_class])
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

    return len(classnames), stats, has_groundtruth, predictions


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
    class_count, stats, has_groundtruth, predictions = classification(model, data_loader_test, bert_model, args=args, device=device)

    if utils.get_rank() == 0:
        save_predictions(args, predictions)
        if has_groundtruth:
            eval(class_count, stats)
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
