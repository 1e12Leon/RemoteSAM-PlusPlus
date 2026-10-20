import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import json

import torch
import torch.utils.data
import utils
from pycocotools.coco import COCO

from data.DiverseDataset import DiverseDataset
from tasks.code.model import RemoteSAM
from tasks.code.runtime import gather_object, load_model_and_text_encoder

import warnings
warnings.filterwarnings("ignore")


def evaluate(gtjson, predjson):
    from tasks.code.metric.cidereval.eval import CIDErEvalCap

    coco_gt = COCO(gtjson)
    coco_dt = coco_gt.loadRes(predjson)
    coco_eval = CIDErEvalCap(coco_gt, coco_dt)
    coco_eval.evaluate()

    print("\n########## Evaluation Summary ##########")
    print("CIDEr:\t{:.3f}%".format(coco_eval.eval["CIDEr"] * 100.0))
    print()


def deduplicate_outputs(gt_json, pred_json):
    images_by_id = {}
    refs_by_image = {}
    preds_by_id = {}

    for image in gt_json["images"]:
        images_by_id.setdefault(image["id"], image)
    for ann in gt_json["annotations"]:
        refs_by_image.setdefault(ann["image_id"], [])
        if ann["caption"] not in refs_by_image[ann["image_id"]]:
            refs_by_image[ann["image_id"]].append(ann["caption"])
    for pred in pred_json:
        preds_by_id.setdefault(pred["image_id"], pred["caption"])

    image_ids = sorted(set(images_by_id) & set(preds_by_id))
    gt_out = {"images": [], "annotations": []}
    pred_out = []

    for image_id in image_ids:
        gt_out["images"].append(images_by_id[image_id])
        pred_out.append({"image_id": image_id, "caption": preds_by_id[image_id]})
        for caption in refs_by_image.get(image_id, []):
            gt_out["annotations"].append({
                "id": len(gt_out["annotations"]),
                "image_id": image_id,
                "caption": caption,
            })

    return gt_out, pred_out


def infer(model, data_loader, bert_model, args, device):
    model = RemoteSAM(model, device, use_EPOC=args.EPOC, args=args, text_encoder=bert_model, EPOC_threshold=0.25)
    metric_logger = utils.MetricLogger(delimiter="  ")

    gt_json = {"images": [], "annotations": []}
    pred_json = []
    classnames = data_loader.dataset.classes

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, "Test:"):
            origin_image, groundtruth, image_name = data

            groundtruth = [g[0] for g in groundtruth]
            image_name = image_name[0]
            image_id = data_loader.dataset.image2id[image_name]

            result = model.captioning(image=origin_image.squeeze(0).numpy(), classnames=classnames)

            pred_json.append({
                "image_id": image_id,
                "caption": result,
            })

            gt_json["images"].append({
                "id": image_id,
                "file_name": image_name,
                "height": origin_image.shape[1],
                "width": origin_image.shape[2],
            })
            for caption in groundtruth:
                gt_json["annotations"].append({
                    "id": len(gt_json["annotations"]),
                    "image_id": image_id,
                    "caption": caption,
                })

            del groundtruth, origin_image, image_name

    gathered = gather_object(gt_json["images"])
    gt_json["images"] = sum(gathered, [])
    gathered = gather_object(gt_json["annotations"])
    gt_json["annotations"] = sum(gathered, [])
    gathered = gather_object(pred_json)
    pred_json = sum(gathered, [])

    if args.local_rank == 0:
        gt_json, pred_json = deduplicate_outputs(gt_json, pred_json)

        gt_path = os.path.join(args.save_path, "groundtruth.json")
        pred_path = os.path.join(args.save_path, "result.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(pred_json, f, indent=4)
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_json, f, indent=4)

        evaluate(gt_path, pred_path)


def main(args):
    utils.init_distributed_mode(args)
    os.makedirs(args.save_path, exist_ok=True)

    dataset_test = DiverseDataset(args)
    if utils.is_dist_avail_and_initialized():
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test,
            num_replicas=utils.get_world_size(),
            rank=utils.get_rank(),
            shuffle=False,
        )
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
    )

    model, bert_model, device = load_model_and_text_encoder(
        args, distributed=utils.is_dist_avail_and_initialized()
    )
    infer(model, data_loader_test, bert_model, args=args, device=device)


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    main(args)
