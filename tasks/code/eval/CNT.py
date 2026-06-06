import sys
import os
# set working directory to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
import torch.utils.data
import utils
from PIL import Image
from data.DiverseDataset import DiverseDataset
from tasks.code.model import RemoteSAM
from tasks.code.runtime import load_model_and_text_encoder
import warnings
warnings.filterwarnings("ignore")


def infer(model, data_loader, bert_model, args, device):
    model = RemoteSAM(model, device, use_EPOC=args.EPOC, args=args, text_encoder=bert_model, EPOC_threshold = 0.15)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_correct, cum_total = 0, 0
    cum_boxes = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, 'Test:'):
            origin_image, groundtruth, image_name, classname, foo = data

            groundtruth = groundtruth.numpy()[0]
            image_name = image_name[0]
            foo = foo.numpy()[0]
            classname = classname[0]

            boxes = model.detection(image=origin_image.squeeze(0).numpy(), classnames=[classname])

            if boxes[classname]:
                boxes = boxes[classname]
                # prior_area_info filter
                indice = []
                for i in range(len(boxes)):
                    x1, y1, x2, y2, conf = boxes[i]
                    area = (x2-x1)*(y2-y1)
                    if area <= foo * 0.8:
                        continue
                    indice.append(i)
                boxes = [boxes[i] for i in indice]
                cnt = len(boxes)
            else:
                cnt = 0 
                
            # evaluate
            if cnt == groundtruth:
                cum_correct += 1

            cum_total += 1

            del groundtruth, origin_image, image_name, foo, boxes

            #! only for debug
            # if cum_total == 10:
            #     break

    # sync ddp processes
    if utils.is_dist_avail_and_initialized():
        counts = torch.tensor([cum_correct, cum_total], dtype=torch.long, device=device)
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        cum_correct = int(counts[0].item())
        cum_total = int(counts[1].item())
    
    # sumarize evaluation
    print("\n########## Evaluation Summary ##########")
    print("Total count: {}".format(cum_total))
    print("Correct count: {}".format(cum_correct))
    print("Accuracy: {:.2f}%".format(cum_correct/cum_total * 100.))
    print()


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
