import sys
import os
# set working directory to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
import torch.utils.data
import utils
import numpy as np
from tasks.code._adapters.datasets_recovered import DiverseDataset
from tasks.code.model_auto import RemoteSAM
from tasks.code.runtime import load_model_and_text_encoder
import warnings
warnings.filterwarnings("ignore")


def infer(model, data_loader, bert_model, args, device):
    model = RemoteSAM(model, device, use_EPOC=args.EPOC, args=args, text_encoder=bert_model, EPOC_threshold = 0.25)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_total = 0
    cum_iou_sum = 0.0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros_like(eval_seg_iou_list, dtype=np.int32)

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, 'Test:'):
            origin_image, groundtruth, image_name, sentence = data

            # prepare
            groundtruth = groundtruth.squeeze(0).numpy()
            image_name = image_name[0]
            sentence = sentence[0]

            boxes = model.visual_grounding(image=origin_image.squeeze(0).numpy(), sentence=sentence)
            IoU = 0.0

            if boxes:
                IoU = computeIoU(boxes, groundtruth)
            cum_iou_sum += IoU

            # evaluate
            for n_eval_iou in range(len(eval_seg_iou_list)):
                if IoU >= eval_seg_iou_list[n_eval_iou]:
                    seg_correct[n_eval_iou] += 1

            cum_total += 1

            del groundtruth, origin_image, image_name, boxes

            #! only for debug
            # if cum_total == 10:
            #     break

    # sync ddp processes
    if utils.is_dist_avail_and_initialized():
        stats = torch.tensor(
            [cum_iou_sum, float(cum_total), *seg_correct.astype(np.float64).tolist()],
            dtype=torch.float64,
            device=device,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        cum_iou_sum = float(stats[0].item())
        cum_total = int(stats[1].item())
        seg_correct = stats[2:].cpu().numpy()

    # sumarize evaluation
    mean_iou = cum_iou_sum / max(cum_total, 1) * 100.0
    test_accu = seg_correct[0] / max(cum_total, 1) * 100.0
    print("\n########## Evaluation Summary ##########")
    print("Total count: {}".format(cum_total))
    print('test-accu:\t{:.2f}%'.format(test_accu))
    print('mIoU:\t{:.2f}%'.format(mean_iou))
    for n_eval_iou in range(len(eval_seg_iou_list)):
        print('precision@{}:\t{:.2f}%'.format(eval_seg_iou_list[n_eval_iou], seg_correct[n_eval_iou] / max(cum_total, 1) * 100.))
    print()


def computeIoU(pred, gt):
    pred_x1, pred_y1, pred_x2, pred_y2 = pred
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    # calculate intersection
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # calculate union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area

    if union_area == 0:
        IoU = 0.0
    else:
        IoU = inter_area / union_area

    return IoU


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
    infer(model, data_loader_test, bert_model, args=args, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)
