import sys
import os
# set working directory to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
import torch.utils.data
import utils
from PIL import Image
import json
from pycocotools.coco import COCO
from data.DiverseDataset import DiverseDataset
from tasks.code.model import RemoteSAM
from tasks.code.runtime import gather_object, load_model_and_text_encoder
import warnings
warnings.filterwarnings("ignore")


def evaluate(gtjson, predjson):
    from tasks.code.metric.cidereval.eval import CIDErEvalCap

    cocoGt = COCO(gtjson)
    cocoDt = cocoGt.loadRes(predjson)
    cocoeval_cap = CIDErEvalCap(cocoGt, cocoDt)
    cocoeval_cap.evaluate()

    print("\n########## Evaluation Summary ##########")
    print("CIDEr:\t{:.3f}%".format(cocoeval_cap.eval['CIDEr'] * 100.))
    print()


def infer(model, data_loader, bert_model, args, device):
    model = RemoteSAM(model, device, use_EPOC=args.EPOC, args=args, text_encoder=bert_model, EPOC_threshold = 0.25)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_total = 0
    json_file = 'result.json'
    gt_json = {"images": [], "annotations": []}
    pred_json = []
    classnames = data_loader.dataset.classes

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, 'Test:'):
            origin_image, groundtruth, image_name = data

            groundtruth = [g[0] for g in groundtruth]
            image_name = image_name[0]

            result = model.captioning(image=origin_image.squeeze(0).numpy(), classnames=classnames)
            # result = result.split(".")[0] + '.'

            # to form json text
            pred_json.append({
                "image_id": data_loader.dataset.image2id[image_name],
                "caption": result
            })

            # to form groundtruth json
            gt_json['images'].append({
                "id": data_loader.dataset.image2id[image_name],
                "file_name": image_name,
                "height": origin_image.shape[1],
                "width": origin_image.shape[2]
            })
            for caption in groundtruth:
                gt_json['annotations'].append({ 
                    "id": len(gt_json['annotations']), 
                    "image_id": data_loader.dataset.image2id[image_name], 
                    "caption": caption
                })

            cum_total += 1

            del groundtruth, origin_image, image_name

            #! only for debug
            # if cum_total == 100:
            #     break

    # sync ddp processes
        # gt_json['images']
    gathered = gather_object(gt_json['images'])
    gt_json['images'] = sum(gathered, []) 
        # gt_json['annotations']
    gathered = gather_object(gt_json['annotations'])
    gt_json['annotations'] = sum(gathered, []) 
        # pred_json
    gathered = gather_object(pred_json)
    pred_json = sum(gathered, []) 

    if args.local_rank == 0:
        gt_path = os.path.join(args.save_path, 'groundtruth.json')
        pred_path = os.path.join(args.save_path, 'result.json')
        # save caption
        pred_json.sort(key=lambda x: x['image_id'])
        with open(pred_path, 'w') as f:
            json.dump(pred_json, f, indent=4)
        # save groundtruth
        gt_json['images'].sort(key=lambda x: x['id'])
        gt_json['annotations'].sort(key=lambda x: x['image_id'])
        with open(gt_path, 'w') as f:
            json.dump(gt_json, f, indent=4)

        evaluate(gt_path, pred_path)


def main(args):
    utils.init_distributed_mode(args)
    os.makedirs(args.save_path, exist_ok=True)

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
    infer(model, data_loader_test, bert_model, args=args, device=device)

    


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)
