"""
Mixup and CutMix 数据增强工具
专门用于分类任务，支持二分类和四分类
"""

import numpy as np
import torch
import torch.nn.functional as F


def mixup_data(x, y, alpha=0.8, device='cuda'):
    """
    Mixup数据增强
    Args:
        x: 输入图像 [B, C, H, W]
        y: 标签 [B]
        alpha: mixup参数
        device: 设备
    Returns:
        mixed_x: 混合后的图像
        y_a, y_b: 原始标签
        lam: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    CutMix数据增强
    Args:
        x: 输入图像 [B, C, H, W]
        y: 标签 [B]
        alpha: cutmix参数
        device: 设备
    Returns:
        mixed_x: 混合后的图像
        y_a, y_b: 原始标签
        lam: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    y_a, y_b = y, y[index]
    
    # 生成随机裁剪框
    W, H = x.size()[2], x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 随机选择裁剪框的中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 计算裁剪框的边界
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # 应用CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # 调整lambda值
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup损失函数
    Args:
        criterion: 原始损失函数
        pred: 模型预测
        y_a, y_b: 混合标签
        lam: 混合系数
    Returns:
        loss: 混合损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def apply_mixup_cutmix(image_batch, label_batch, num_classes_batch, args, device):
    """
    应用Mixup或CutMix数据增强，智能处理混合类别
    Args:
        image_batch: 输入图像批次 [B, C, H, W]
        label_batch: 标签批次 [B]
        num_classes_batch: 每个样本的类别数 [B]
        args: 训练参数
        device: 设备
    Returns:
        image_batch: 增强后的图像
        label_batch_a: 原始标签A
        label_batch_b: 原始标签B
        lam: 混合系数
        augmentation_type: 增强类型 (0: 无, 1: Mixup, 2: CutMix)
    """
    augmentation_type = 0  # 0: 无增强, 1: Mixup, 2: CutMix
    lam = 1.0
    label_batch_a = label_batch
    label_batch_b = label_batch
    
    # 检查是否启用数据增强
    if hasattr(args, 'mixup_alpha') and args.mixup_alpha > 0 and hasattr(args, 'mixup_prob') and args.mixup_prob > 0:
        if np.random.random() < args.mixup_prob:
            # 决定使用Mixup还是CutMix
            if hasattr(args, 'cutmix_alpha') and args.cutmix_alpha > 0 and hasattr(args, 'mixup_switch_prob'):
                if np.random.random() < args.mixup_switch_prob:
                    # 使用CutMix
                    image_batch, label_batch_a, label_batch_b, lam = cutmix_data(
                        image_batch, label_batch, args.cutmix_alpha, device
                    )
                    augmentation_type = 2
                else:
                    # 使用Mixup
                    image_batch, label_batch_a, label_batch_b, lam = mixup_data(
                        image_batch, label_batch, args.mixup_alpha, device
                    )
                    augmentation_type = 1
            else:
                # 只使用Mixup
                image_batch, label_batch_a, label_batch_b, lam = mixup_data(
                    image_batch, label_batch, args.mixup_alpha, device
                )
                augmentation_type = 1
    
    return image_batch, label_batch_a, label_batch_b, lam, augmentation_type


def validate_labels(labels, num_classes, task_name="classification"):
    """
    验证标签是否在有效范围内
    Args:
        labels: 标签张量
        num_classes: 类别数
        task_name: 任务名称（用于错误信息）
    """
    if labels.dim() == 0:  # 标量
        labels = labels.unsqueeze(0)
    
    # 检查标签类型
    if not labels.dtype in [torch.long, torch.int64]:
        labels = labels.long()
    
    # 检查标签范围
    min_label = labels.min().item()
    max_label = labels.max().item()
    
    if min_label < 0 or max_label >= num_classes:
        raise ValueError(
            f"Labels in {task_name} task must be in range [0, {num_classes-1}], "
            f"but got range [{min_label}, {max_label}]"
        )
    
    return labels


def compute_mixup_loss(outputs, labels_a, labels_b, lam, criterion, num_classes, task_name="classification"):
    """
    计算Mixup损失
    Args:
        outputs: 模型输出
        labels_a, labels_b: 混合标签
        lam: 混合系数
        criterion: 损失函数
        num_classes: 类别数
        task_name: 任务名称
    Returns:
        loss: 混合损失
    """
    # 验证标签
    labels_a = validate_labels(labels_a, num_classes, f"{task_name}_labels_a")
    labels_b = validate_labels(labels_b, num_classes, f"{task_name}_labels_b")
    
    # 计算混合损失
    return mixup_criterion(criterion, outputs, labels_a, labels_b, lam)


def log_augmentation_info(writer, iteration, augmentation_type, lam, task_name="cls"):
    """
    记录数据增强信息到TensorBoard
    Args:
        writer: TensorBoard writer
        iteration: 当前迭代次数
        augmentation_type: 增强类型
        lam: 混合系数
        task_name: 任务名称
    """
    writer.add_scalar(f'info/{task_name}_augmentation_type', augmentation_type, iteration)
    if augmentation_type > 0:
        if augmentation_type == 1:  # Mixup
            writer.add_scalar(f'info/{task_name}_mixup_lam', lam, iteration)
        elif augmentation_type == 2:  # CutMix
            writer.add_scalar(f'info/{task_name}_cutmix_lam', lam, iteration)