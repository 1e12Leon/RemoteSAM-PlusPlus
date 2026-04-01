import datetime
import os
import time
import torch
import torch.utils.data
import swanlab
swanlab.login(api_key='fvHEgPt2nOueccBWkJFtl')
import cv2
import random
import transforms as T
import utils
import numpy as np
import gc
import operator
from functools import reduce
from bert.modeling_bert import BertModel
from lib import segmentation
from loss.loss import Loss

def seed_everything(seed=2401):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset(image_set, transform, args, eval):
    # from data.dataset_refer_bert import All_Dataset, ReferDataset
    from data.all_dataset import All_Dataset
    ds = All_Dataset(args,
                      split=image_set,
                      max_tokens=20,
                      image_transforms=transform,
                      eval_mode=eval
                      )
    num_classes = 2

    return ds, num_classes


def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union


def get_transform(args):
    transforms = [
                  T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]
    return T.Compose(transforms)


def criterion(input, target, weight=0.1):
    return Loss(weight=weight)(input, target)


def evaluate(model, data_loader, bert_model, epoch):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    cum_I, cum_U = 0, 0 
    cum_I_rgb, cum_U_rgb = 0, 0
    cum_I_sar, cum_U_sar = 0, 0
    
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    total_loss = 0
    total_its = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions, exist, modal = data
            # convert modal to tensor int {0,1}
            if isinstance(modal, (list, tuple)):
                if len(modal) > 0 and isinstance(modal[0], str):
                    modal = [1 if ("sar" in m.lower()) else 0 for m in modal]
            else:
                if isinstance(modal, str):
                    modal = 1 if ("sar" in modal.lower()) else 0
            modal = torch.as_tensor(modal, dtype=torch.long, device=image.device)
            pixels = cv2.countNonZero(target.data.numpy()[0]) / 230400.
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                output = model(image, embedding, l_mask=attentions, modal=modal)
            else:
                output = model(image, sentences, l_mask=attentions, modal=modal)

            loss = criterion(output, target)
            total_loss += loss.item()

            preds = output.argmax(1)
            
            current_batch_size = preds.size(0)
            
            for b in range(current_batch_size):
                pred_sample = preds[b]   
                gt_sample = target[b]   
                modal_sample = modal[b] 
                
                intersection = torch.sum(pred_sample * gt_sample).item()

                union = torch.sum(pred_sample) + torch.sum(gt_sample) - intersection
                union = union.item()

                if union == 0:

                    sample_iou = 0.0
                else:
                    sample_iou = float(intersection) / float(union)
                
                mean_IoU.append(sample_iou)
                
                cum_I += intersection
                cum_U += union
                
                modal_str = str(modal_sample).lower().strip()

                modal_id = int(modal_sample.item())
                
                if modal_id == 0:
                    cum_I_rgb += intersection
                    cum_U_rgb += union
                else:
                    cum_I_sar += intersection
                    cum_U_sar += union
                
                # 统计 Precision@k
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (sample_iou >= eval_seg_iou)
                
                seg_total += 1

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU) if len(mean_IoU) > 0 else 0.0

    oiou_total = cum_I * 100. / cum_U if cum_U > 0 else 0.0
    oiou_rgb = cum_I_rgb * 100. / cum_U_rgb if cum_U_rgb > 0 else 0.0
    oiou_sar = cum_I_sar * 100. / cum_U_sar if cum_U_sar > 0 else 0.0

    print('Final results:')
    print('Mean IoU is %.2f' % (mIoU * 100.))
    
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    
    results_str += '    Overall IoU (Total): %.2f\n' % oiou_total
    results_str += '    Overall IoU (RGB)  : %.2f\n' % oiou_rgb
    results_str += '    Overall IoU (SAR)  : %.2f\n' % oiou_sar
    print(results_str)

    if args.local_rank == 0:
        swanlab.log({
            "val mIoU": mIoU * 100., 
            "val oiou": oiou_total,
            "val oiou_rgb": oiou_rgb,
            "val oiou_sar": oiou_sar,
            "val Loss": total_loss / total_its
        })

    return mIoU * 100., oiou_total


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    # for data in data_loader:
    for i, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        total_its += 1
        image, target, sentences, attentions, exist, modal = data
        # convert modal to tensor int {0,1}
        if isinstance(modal, (list, tuple)):
            if len(modal) > 0 and isinstance(modal[0], str):
                modal = [1 if ("sar" in m.lower()) else 0 for m in modal]
        else:
            if isinstance(modal, str):
                modal = 1 if ("sar" in modal.lower()) else 0
        modal = torch.as_tensor(modal, dtype=torch.long, device=image.device)

        image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:

            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            output = model(image, embedding, attentions, modal=modal)#, sentences_hidden_state)# [4,2,120,120]
        else:
            output = model(image, sentences, attentions, modal=modal)#, sentences_hidden_state)
        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()


        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        # gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

    if args.local_rank == 0:
        swanlab.log({
            "Train Loss": train_loss / total_its,})


def main(args):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args, eval=False)

    dataset_test, _ = get_dataset("test",
                                  get_transform(args=args),
                                  args=args, eval=True)
    
    # print(dataset_test)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module  #ddp

    # print(model)
    if args.model != 'lavt_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model
    else:
        bert_model = None
        single_bert_model = None

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        # 修改后（打印不匹配的键，检查是否有关键层未加载）：
        missing_keys, unexpected_keys = single_model.load_state_dict(checkpoint['model'], strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if not m.requires_grad:
            continue
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'lavt_one':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.module.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']

    else:
        resume_epoch = -999

    # training loops
    # if args.local_rank == 0:
    #     swanlab.watch(model, log="all")


    for epoch in range(max(0, resume_epoch+1), args.epochs):
    # for epoch in range(0, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model)
        iou, overallIoU = evaluate(model, data_loader_test, bert_model, epoch)
        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        best = (best_oIoU < overallIoU)
        if single_bert_model is not None:
            dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}
        else:
            dict_to_save = {'model': single_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}

        if best:
            print('Better epoch: {}\n'.format(epoch))
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU
        utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                        'model_last_{}.pth'.format(args.model_id)))
        # if args.local_rank == 0:
        #     swanlab.save('model.h5')

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    seed_everything()
    parser = get_parser()
    args = parser.parse_args()
    if args.local_rank == 0:
        swanlab.init(project="remotesam_sar_toy_experiment_2000")
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)