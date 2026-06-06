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
from tasks.code.runtime import gather_object, load_model_and_text_encoder
import warnings
warnings.filterwarnings("ignore")


def infer(model, data_loader, bert_model, args, device):
    model = RemoteSAM(model, device, use_EPOC=args.EPOC, args=args, text_encoder=bert_model, EPOC_threshold = 0.25)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_total = 0
    cum_boxes = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, 'Test:'):
            origin_image, groundtruth, image_name = data

            groundtruth = groundtruth.squeeze(0).numpy()
            image_name = image_name[0]

            boxes = model.detection(image=origin_image.squeeze(0).numpy(), classnames=[args._class])
            if boxes[args._class]:
                boxes = boxes[args._class]

                # to form txt
                if len(boxes) > 0:
                    for i in range(len(boxes)):
                        temp = boxes[i]
                        temp.append(image_name)
                        cum_boxes.append(temp)

            cum_total += 1

            del groundtruth, origin_image, image_name, boxes

            #! only for debug
            # if cum_total == 10:
            #     break

    # sync ddp processes
        # cum_boxes
    gathered = gather_object(cum_boxes)
    cum_boxes = sum(gathered, []) 

    cum_boxes.sort(key=lambda x: x[5])

    if args.local_rank == 0:
        # save result
        os.makedirs(args.save_path, exist_ok=True)
        result_path = os.path.join(args.save_path, 'Task2_' + args._class + '.txt')
        with open(result_path, 'w') as f:
            for box in cum_boxes:
                x1, y1, x2, y2, conf, image_name = box
                pts = [x1, y1, x2, y1, x2, y2, x1, y2]
                box_string = ' '.join([str(float(a)) for a in pts])
                f.write(image_name + " " + str(conf) + " " + box_string + '\n')

    print("Done evaluating the category: ", args._class)
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
