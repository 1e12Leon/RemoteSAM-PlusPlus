import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
import torch.utils.data
import utils
import numpy as np
import transforms as T
import random
from tasks.code.runtime import build_ref_dataset, load_model_and_text_encoder


def get_dataset(image_set, transform, args):
    return build_ref_dataset(args, image_set, transform), 2


def evaluate(model, data_loader, bert_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    with torch.no_grad():

        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions, _, modal = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            modal = modal.to(device=device, dtype=torch.long).reshape(-1)
            target = target.cpu().data.numpy()
            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1).to(dtype=image.dtype)
                output = model(image, embedding, l_mask=attentions.unsqueeze(-1), modal=modal)
            else:
                output = model(image, sentences, l_mask=attentions, modal=modal)

            output = output.cpu()

            output_mask = output.argmax(1).data.numpy()

            I, U = computeIoU(output_mask, target)
            if U == 0:
                this_iou = 0.0
            else:
                this_iou = I*1.0/U
            mean_IoU.append(this_iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)

            seg_total += 1


            del image, target, sentences, attentions, output,output_mask
            if bert_model is not None:
                del last_hidden_states, embedding

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)




def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):

    # 1. 设置运行设备 (CPU or GPU)
    device = torch.device(args.device)

    # 2. 加载数据集
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    # 3. 初始化模型并加载当前项目 checkpoint
    model, bert_model, device = load_model_and_text_encoder(args, distributed=False)

    # 6. 调用评测函数
    evaluate(model, data_loader_test, bert_model, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
