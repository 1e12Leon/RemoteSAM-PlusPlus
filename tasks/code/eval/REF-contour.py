import torch
import torch.utils.data
import utils
import numpy as np
import transforms as T
import random
from bert.modeling_bert import BertModel
from lib import segmentation
import cv2
from tqdm import tqdm
from contour.cal_edge import calculate_metrics_from_tensors


def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def evaluate(model, data_loader, bert_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

   # 存储每个阈值下的平均指标
    precise_list_per_thresh = []
    recall_list_per_thresh = []
    f_mean_per_thresh = []
    liou3_per_thresh = []
    liou5_per_thresh = []
    max_dist = 0.0075 

    print("Step 1/2: Running model inference for all (image, sentence) pairs...")
    all_prob_maps = []
    all_gt_masks = []
    with torch.no_grad():

        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                output = output.cpu()
                # 获取前景 (class 1) 的概率图
                prob_map = torch.nn.functional.softmax(output, dim=1)[:, 1, :, :] # Shape: [1, H, W]
                # 保存概率图和 GT 图
                all_prob_maps.append(prob_map.squeeze(0).cpu())
                all_gt_masks.append(target.squeeze(0).cpu())

                del image, target, sentences, attentions, output, prob_map
                if bert_model is not None:
                    del last_hidden_states, embedding

    # 2. 第二步：遍历阈值，在所有已保存的 (概率图, 真值图) 对上计算指标
    print(f"\nStep 2/2: Calculating metrics across thresholds for {len(all_prob_maps)} total samples...")
    num_samples = len(all_prob_maps)
    
    # per_sample_scores 用于 OIS 计算，记录每个样本在所有阈值下的 F-score
    per_sample_scores = [[] for _ in range(num_samples)]

    for i in tqdm(range(1, 100), desc="Thresholds"):
        threshold = i / 100.0
        
        # 临时存储当前阈值下所有样本的结果
        recalls_current_thresh = []
        precisions_current_thresh = []
        fscores_current_thresh = []
        liou3s_current_thresh = []
        liou5s_current_thresh = []

        # 遍历所有样本对
        for idx in range(num_samples):
            prob_map = all_prob_maps[idx].to(device)
            gt_mask = all_gt_masks[idx].to(device)

            # 根据阈值生成二值预测掩码
            pred_mask = (prob_map >= threshold).long()

            # 调用函数计算指标
            precision, recall, f_score, liou3, liou5 = calculate_metrics_from_tensors(
                pred_mask,
                gt_mask,
                max_dist=max_dist,
                device=device
            )

            # 收集当前样本、当前阈值的指标
            recalls_current_thresh.append(recall)
            precisions_current_thresh.append(precision)
            fscores_current_thresh.append(f_score)
            liou3s_current_thresh.append(liou3)
            liou5s_current_thresh.append(liou5)
            
            # 为 OIS 收集数据：记录这个样本在这个阈值下的 F-score
            per_sample_scores[idx].append(f_score)

        # 计算当前阈值下所有样本的平均指标
        avg_recall = np.mean(recalls_current_thresh) if recalls_current_thresh else 0
        avg_precision = np.mean(precisions_current_thresh) if precisions_current_thresh else 0
        avg_fscore = np.mean(fscores_current_thresh) if fscores_current_thresh else 0
        avg_liou3 = np.mean(liou3s_current_thresh) if liou3s_current_thresh else 0
        avg_liou5 = np.mean(liou5s_current_thresh) if liou5s_current_thresh else 0
        
        # 为 ODS 收集数据：将当前阈值的平均指标存入列表
        recall_list_per_thresh.append(avg_recall)
        precise_list_per_thresh.append(avg_precision)
        f_mean_per_thresh.append(avg_fscore)
        liou3_per_thresh.append(avg_liou3)
        liou5_per_thresh.append(avg_liou5)

    # 3. 第三步：计算最终的 ODS 和 OIS
    # =======================================================================
    print("\nFinal Results Calculation...")
    # ODS: max of (mean F-score over all samples)
    ODS = np.max(f_mean_per_thresh) if f_mean_per_thresh else 0.0
    
    # OIS: mean of (max F-score over thresholds for each sample)
    per_sample_max_f = [np.max(f_list) if f_list else 0.0 for f_list in per_sample_scores]
    OIS = np.mean(per_sample_max_f) if per_sample_max_f else 0.0

    # 找到 ODS 对应的各项指标
    ods_index = np.argmax(f_mean_per_thresh) if f_mean_per_thresh else -1
    best_threshold = (ods_index + 1) / 100.0
    ods_precision = precise_list_per_thresh[ods_index] if ods_index != -1 else 0.0
    ods_recall = recall_list_per_thresh[ods_index] if ods_index != -1 else 0.0
    ods_liou3 = liou3_per_thresh[ods_index] if ods_index != -1 else 0.0
    ods_liou5 = liou5_per_thresh[ods_index] if ods_index != -1 else 0.0

    print("\n" + "="*40)
    print("               Final Metrics")
    print("="*40)
    print(f"Total evaluation samples (image-sentence pairs): {num_samples}")
    print(f"ODS (Optimal Dataset Scale) F-score: {ODS:.4f}")
    print(f"OIS (Optimal Image Scale) F-score:   {OIS:.4f}")
    print("-"*40)
    print(f"Metrics at best threshold ({best_threshold:.2f}) for ODS:")
    print(f"    - Precision: {ods_precision:.4f}")
    print(f"    - Recall:    {ods_recall:.4f}")
    print(f"    - LineIoU@3: {ods_liou3:.4f}")
    print(f"    - LineIoU@5: {ods_liou5:.4f}")
    print("="*40)


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
    # 3. 初始化视觉分割模型
    single_model = segmentation.__dict__[args.model](pretrained='',args=args)

    # 4. 加载预训练权重
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    single_model.load_state_dict(checkpoint['model'], strict=False) # 将权重加载到模型中
    model = single_model.to(device)

    # 5. 初始化文本编码器 BERT 模型
    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    # 6. 调用评测函数
    evaluate(model, data_loader_test, bert_model, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
