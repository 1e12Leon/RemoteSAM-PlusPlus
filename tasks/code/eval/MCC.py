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


def eval(class_count, correct_total):
    correct, total = correct_total.cpu().double()
    accuracy = correct / (total + 1e-15)

    # sumarize evaluation
    print("\n########## Evaluation Summary ##########")
    print("Accuracy: {:.3f}%".format(accuracy.item() * 100.))
    print()


def classification(model, data_loader, bert_model, args, device):
    model = RemoteSAM(model, device, use_EPOC=args.EPOC, args=args, text_encoder=bert_model, EPOC_threshold = 0.25)
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    classnames = data_loader.dataset.classes
    correct_total = torch.zeros(2, dtype=torch.long, device=device)

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, 'Test:'):
            origin_image, groundtruth, image_name = data

            groundtruth = groundtruth.item()
            image_name = image_name[0]

            _, prob_ins = model.multi_class_cls(image=origin_image.squeeze(0).numpy(), classnames=classnames, return_prob=True)

            pred = max(classnames, key=lambda classname: prob_ins[classname])
            pred_index = classnames.index(pred)
            correct_total[0] += int(pred_index == groundtruth)
            correct_total[1] += 1

            del groundtruth, origin_image, image_name, prob_ins

            #! only for debug
            # if cum_total == 10:
            #     break

    if utils.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(correct_total, op=torch.distributed.ReduceOp.SUM)

    return len(classnames), correct_total


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
    class_count, correct_total = classification(model, data_loader_test, bert_model, args=args, device=device)

    if utils.get_rank() == 0:
        eval(class_count, correct_total)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)
