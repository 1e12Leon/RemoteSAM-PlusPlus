import cv2
import numpy as np
# import cupy as cp
import torch
import os
import torch.nn.functional as F
import torch.nn as nn
# np.set_printoptions(threshold=np.inf)
from torchvision.transforms import Resize
import math
from PIL import Image


def thicken_edges(image, thickness, device):
    image = image.float()
    """
    Image has to be tesnor, shape like [h,w]
    """
    # 读取图像并转换为 PyTorch 张量
    thickness = int(thickness)
    image = image.unsqueeze(0).unsqueeze(0)  # 将图像扩展为形状 [1, 1, H, W]

    # 根据加粗的像素数计算卷积核大小
    kernel_size = 2 * thickness + 1  # 例如，厚度为2时，卷积核大小为5x5
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
    kernel = kernel.to(device)
    # 使用卷积操作来模拟膨胀（加粗）效果
    padding = thickness

    dilated_image = F.conv2d(image, kernel, padding=padding)

    # 将结果张量转换回图像格式
    dilated_image = dilated_image.squeeze(0).squeeze(0)
    return dilated_image


def ceil_to_even(n):
    # 先对浮点数上取整为整数
    n_ceil = math.ceil(n)
    # 检查是否为偶数
    return n_ceil if n_ceil % 2 == 0 else n_ceil + 1


def line_IOU(gt, infer_result, bolden, device):
    """
    Input Shape is like [1024 1024]
    """
    gt_bolden = thicken_edges(gt, bolden, device=device)
    infer_result_bolden = thicken_edges(infer_result, bolden, device=device)

    intersection = torch.logical_and(infer_result_bolden, gt_bolden).sum().item()
    union = torch.logical_or(infer_result_bolden, gt_bolden).sum().item()

    iou = intersection / union if union > 0 else 0.0

    return iou


def calculate_one_image_tensor_origin(max_dist, model_edge_path, ground_truth_path, threshold, nms):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    if nms:
        model_edge = cv2.imread(model_edge_path, cv2.IMREAD_UNCHANGED) / 255.0
    else:
        model_edge = np.load(model_edge_path)
    contour_binary_map = np.where(model_edge > threshold, 1, 0).astype(np.uint8)
    model_edge_b_t = torch.tensor(contour_binary_map)
    model_edge_b_t = model_edge_b_t.to(device)

    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_UNCHANGED) / 255.0
    ground_truth = cv2.resize(ground_truth, (model_edge_b_t.shape[0], model_edge_b_t.shape[1]),
                              interpolation=cv2.INTER_NEAREST)
    ground_truth = np.where(ground_truth > 0.1, 1, 0)

    gt_t = torch.tensor(ground_truth)
    gt_t = gt_t.to(device)

    tolerance = model_edge_b_t.shape[0] * max_dist
    tolerance = ceil_to_even(tolerance)
    bolden_thickness = tolerance / 2

    gt_b_t = thicken_edges(gt_t, bolden_thickness, device)
    model_edge_b_b_t = thicken_edges(model_edge_b_t, bolden_thickness, device)

    # if torch.all(model_edge_b_t == 0) or torch.all(gt_t == 0) or torch.all(model_edge_b_b_t==0) or torch.all(gt_b_t==0):
    #     return 0,0,0,0,0
    model_edge_b_b_t = (model_edge_b_b_t != 0).float()
    model_edge_b_t = (model_edge_b_t != 0).float()
    gt_b_t = (gt_b_t != 0).float()
    gt_t = (gt_t != 0).float()

    lineIoU_3 = line_IOU(gt_t, model_edge_b_t, 1, device)
    lineIoU_5 = line_IOU(gt_t, model_edge_b_t, 2, device)

    recall_numerator = torch.sum((gt_t != 0) & (model_edge_b_b_t != 0))
    recall_denominator = torch.nonzero(gt_t).size(0)
    recall = recall_numerator / (recall_denominator + 1e-8)
    precise_numerator = torch.sum((model_edge_b_t != 0) & (gt_b_t != 0))
    precise_denominator = torch.nonzero(model_edge_b_t).size(0)
    precise = precise_numerator / (precise_denominator + 1e-8)

    F_score = 2 * precise * recall / (precise + recall + 1e-8)
    # print("Recall:",recall,"Precise:",precise,"F:",F_score)
    return recall.item(), precise.item(), F_score.item(), lineIoU_3, lineIoU_5



def calculate_metrics_from_tensors(pred_mask_tensor, gt_mask_tensor, max_dist, device):
    """
    直接从二值掩码张量计算边缘指标。
    - pred_mask_tensor: 模型的预测二值掩码 (0和1), a torch.Tensor on GPU.
    - gt_mask_tensor: 真实的二值掩码 (0和1), a torch.Tensor on GPU.
    - max_dist: 用于计算 tolerance 的最大距离参数。
    - device: 当前使用的计算设备 (e.g., 'cuda:0').
    """
    # 1. 将分割掩码转换为轮廓/边缘图
    # 首先需要将Tensor转为CPU上的Numpy数组以使用OpenCV
    pred_mask_np = (pred_mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    gt_mask_np = (gt_mask_tensor.cpu().numpy() * 255).astype(np.uint8)

    # 提取轮廓
    pred_contours, _ = cv2.findContours(pred_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, _ = cv2.findContours(gt_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建空白画布并将轮廓画上去，得到边缘图
    pred_edge_np = np.zeros_like(pred_mask_np)
    gt_edge_np = np.zeros_like(gt_mask_np)
    cv2.drawContours(pred_edge_np, pred_contours, -1, 1, 1) # 轮廓像素值为1，厚度为1
    cv2.drawContours(gt_edge_np, gt_contours, -1, 1, 1) # 轮廓像素值为1，厚度为1

    # 将边缘图转回Tensor
    model_edge_b_t = torch.from_numpy(pred_edge_np).to(device)
    gt_t = torch.from_numpy(gt_edge_np).to(device)

    # 2. 复用您原始代码的核心计算逻辑
    if torch.sum(gt_t) == 0 or torch.sum(model_edge_b_t) == 0:
        # 如果真实标签或预测结果中没有边缘，则无法计算，返回0
        return 0.0, 0.0, 0.0, 0.0, 0.0

    tolerance = model_edge_b_t.shape[0] * max_dist
    tolerance = ceil_to_even(tolerance)
    bolden_thickness = tolerance / 2

    gt_b_t = thicken_edges(gt_t, bolden_thickness, device)
    model_edge_b_b_t = thicken_edges(model_edge_b_t, bolden_thickness, device)

    model_edge_b_b_t = (model_edge_b_b_t != 0).float()
    model_edge_b_t = (model_edge_b_t != 0).float()
    gt_b_t = (gt_b_t != 0).float()
    gt_t = (gt_t != 0).float()

    lineIoU_3 = line_IOU(gt_t, model_edge_b_t, 1, device)
    lineIoU_5 = line_IOU(gt_t, model_edge_b_t, 2, device)

    recall_numerator = torch.sum((gt_t != 0) & (model_edge_b_b_t != 0))
    recall_denominator = torch.nonzero(gt_t, as_tuple=False).size(0)
    recall = recall_numerator / (recall_denominator + 1e-8)

    precise_numerator = torch.sum((model_edge_b_t != 0) & (gt_b_t != 0))
    precise_denominator = torch.nonzero(model_edge_b_t, as_tuple=False).size(0)
    precise = precise_numerator / (precise_denominator + 1e-8)

    F_score = 2 * precise * recall / (precise + recall + 1e-8)

    return precise.item(), recall.item(), F_score.item(), lineIoU_3, lineIoU_5