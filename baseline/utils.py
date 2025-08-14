import numpy as np
import torch
from medpy import metric
import torch.nn as nn
import cv2
import os
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for segmentation.
    - inputs: (B, C, H, W) logits or probabilities
    - target: (B, H, W) with class indices in [0, C-1]
    """
    def __init__(self, n_classes=2, gamma=2.0, alpha=None, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.n_classes = n_classes
        self.gamma = gamma
        self.eps = eps

        # alpha: per-class weight for focal loss, shape (C,)
        if alpha is None:
            self.alpha = torch.ones(n_classes, dtype=torch.float32)
        else:
            # list/tuple/tensor -> tensor(C,)
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
            assert self.alpha.numel() == n_classes, "alpha length must equal n_classes"

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, target, weight=None, softmax=False):
        """
        inputs: (B, C, H, W) logits/probs
        target: (B, H, W) int64
        weight: optional per-class weight list/tuple, length C
        softmax: if True, apply softmax to inputs first
        """
        if softmax:
            probs = F.softmax(inputs, dim=1)
        else:
            probs = inputs  # assume already probabilities if softmax=False

        target_1h = self._one_hot_encoder(target).type_as(probs)  # (B, C, H, W)

        if weight is None:
            weight = [1.0] * self.n_classes
        # to device & tensor
        weight = torch.as_tensor(weight, dtype=probs.dtype, device=probs.device)
        alpha = self.alpha.to(device=probs.device, dtype=probs.dtype)

        loss = 0.0
        eps = self.eps

        # 按类别分别计算，再做加权平均（与 DiceLoss 风格一致）
        for c in range(self.n_classes):
            p_c = probs[:, c, ...].clamp(min=eps, max=1.0 - eps)      # (B, H, W)
            mask_c = target_1h[:, c, ...]                              # (B, H, W), {0,1}

            # focal term: -alpha * (1 - p_t)^gamma * log(p_t)
            fl_c = - alpha[c] * torch.pow(1.0 - p_c, self.gamma) * torch.log(p_c)
            # 只在该类的正样本像素上取平均
            denom = mask_c.sum() + eps
            loss_c = (fl_c * mask_c).sum() / denom

            loss = loss + weight[c] * loss_c

        return loss / self.n_classes

class DiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0 and gt.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         return dice, True
#     elif pred.sum() > 0 and gt.sum() == 0:
#         return 0, False
#     elif pred.sum() == 0 and gt.sum() > 0:
#         return 0, True
#     else:
#         return 0, False



def calculate_metric_percase(pred_batch, gt_batch):
    """
    按batch计算Dice系数和有效标记
    输入:
        pred_batch: [B, H, W] 或 [B, H, W, D] 的预测二值化数组
        gt_batch:   同维度的真实标签
    返回:
        dice_list:  每个样本的Dice值列表
        valid_list: 对应样本是否有效的布尔列表
    """
    # 二值化处理
    pred_batch = (pred_batch > 0).astype(np.uint8)
    gt_batch = (gt_batch > 0).astype(np.uint8)
    
    dice_list = []
    valid_list = []
    
    for pred, gt in zip(pred_batch, gt_batch):
        if pred.sum() > 0 and gt.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            dice_list.append(dice)
            valid_list.append(True)
        elif pred.sum() > 0 and gt.sum() == 0:
            dice_list.append(0.0)
            valid_list.append(False)
        elif pred.sum() == 0 and gt.sum() > 0:
            dice_list.append(0.0)
            valid_list.append(True)
        else:
            dice_list.append(0.0)
            valid_list.append(False)
    
    return dice_list, valid_list


def omni_seg_test(image, label, net, classes, ClassStartIndex=1, test_save_path=None, case=None,
                  prompt=False,
                  type_prompt=None,
                  nature_prompt=None,
                  position_prompt=None,
                  task_prompt=None,
                  dataset_name=None
                  ):
    label = label.cpu().detach().numpy()
    image_save = image.cpu().detach().numpy()
    input = image.cuda()
    if prompt:
        position_prompt = position_prompt.cuda()
        task_prompt = task_prompt.cuda()
        type_prompt = type_prompt.cuda()
        nature_prompt = nature_prompt.cuda()
    net.eval()
    with torch.no_grad():
        if prompt:
            seg_out = net((input, position_prompt, task_prompt, type_prompt, nature_prompt))[0]
        else:
            seg_out = net(input)[0]
        out_label_back_transform = torch.cat(
            [seg_out[:, 0:1], seg_out[:, ClassStartIndex:ClassStartIndex+classes-1]], axis=1)
        out = torch.argmax(torch.softmax(out_label_back_transform, dim=1), dim=1)
        prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        image = (image_save - np.min(image_save)) / (np.max(image_save) - np.min(image_save))
        cv2.imwrite(test_save_path + '/'+case + "_pred.png", (prediction*255).astype(np.uint8))
        cv2.imwrite(test_save_path + '/'+case + "_img.png", ((image.squeeze(0))*255).astype(np.uint8))
        cv2.imwrite(test_save_path + '/'+case + "_gt.png", (label.transpose(1, 2, 0)*255).astype(np.uint8))
    return metric_list


def omni_seg_test_TU(image, label, net, classes, ClassStartIndex=1, test_save_path=None, case=None,
                  prompt=False,
                  type_prompt=None,
                  nature_prompt=None,
                  position_prompt=None,
                  task_prompt=None,
                  dataset_name=None
                  ):
    label = label.cpu().detach().numpy()
    image_save = image.cpu().detach().numpy()
    input = image.cuda()
    if prompt:
        position_prompt = position_prompt.cuda()
        task_prompt = task_prompt.cuda()
        type_prompt = type_prompt.cuda()
        nature_prompt = nature_prompt.cuda()
    net.eval()
    with torch.no_grad():
        if prompt:
            seg_out = net(input, position_prompt, task_prompt, type_prompt, nature_prompt)[0]
        else:
            seg_out = net(input)[0]
        out_label_back_transform = torch.cat(
            [seg_out[:, 0:1], seg_out[:, ClassStartIndex:ClassStartIndex+classes-1]], axis=1)
        out = torch.argmax(torch.softmax(out_label_back_transform, dim=1), dim=1)
        prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        image = (image_save - np.min(image_save)) / (np.max(image_save) - np.min(image_save))
        cv2.imwrite(test_save_path + '/'+ dataset_name + '/'+case + "_pred.png", (prediction.transpose(1, 2, 0)*255).astype(np.uint8))
        cv2.imwrite(test_save_path + '/'+ dataset_name + '/'+case + "_img.png", ((image.squeeze(0).transpose(1, 2, 0))*255).astype(np.uint8))
        cv2.imwrite(test_save_path + '/'+ dataset_name + '/'+case + "_gt.png", (label.transpose(1, 2, 0)*255).astype(np.uint8))
    return metric_list


def omni_seg_test_decoders(image, label, net, classes, ClassStartIndex=1, test_save_path=None, case=None,
                  prompt=False,
                  type_prompt=None,
                  nature_prompt=None,
                  position_prompt=None,
                  task_prompt=None,
                  dataset_name=None
                  ):
    label = label.cpu().detach().numpy()
    image_save = image.cpu().detach().numpy()
    input = image.cuda()
    if prompt:
        position_prompt = position_prompt.cuda()
        task_prompt = task_prompt.cuda()
        type_prompt = type_prompt.cuda()
        nature_prompt = nature_prompt.cuda()
    net.eval()
    with torch.no_grad():
        if prompt:
            seg_out = net((input, position_prompt, task_prompt, type_prompt, nature_prompt), use_dataset_specific=True, dataset_name=dataset_name, task_type='seg', num_classes=classes)
        else:
            seg_out = net(input, use_dataset_specific=True, dataset_name=dataset_name, task_type='seg', num_classes=2)
        # out_label_back_transform = torch.cat(
        #     [seg_out[:, 0:1], seg_out[:, ClassStartIndex:ClassStartIndex+classes-1]], axis=1)
        out = torch.argmax(torch.softmax(seg_out, dim=1), dim=1)
        prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        os.makedirs(os.path.join(test_save_path, dataset_name), exist_ok=True)
        
        image = (image_save - np.min(image_save)) / (np.max(image_save) - np.min(image_save))
        print(os.path.join(test_save_path, dataset_name, case.split('.')[0] + "_pred.png"))
        cv2.imwrite(os.path.join(test_save_path, dataset_name, case.split('.')[0] + "_pred.png"), (prediction.transpose(1, 2, 0)*255).astype(np.uint8))
        cv2.imwrite(os.path.join(test_save_path, dataset_name, case.split('.')[0] + "_img.png"), ((image.squeeze(0))*255).astype(np.uint8))
        cv2.imwrite(os.path.join(test_save_path, dataset_name, case.split('.')[0] + "_gt.png"), (label.transpose(1, 2, 0)*255).astype(np.uint8))

        # cv2.imwrite(test_save_path + '/'+dataset_name + '/' + case.split('.')[0] + "_pred.png", (prediction.transpose(1, 2, 0)*255).astype(np.uint8))
        # cv2.imwrite(test_save_path + '/'+dataset_name + '_'+ case.split('.')[0] + "_img.png", ((image.squeeze(0))*255).astype(np.uint8))
        # cv2.imwrite(test_save_path + '/'+dataset_name + '_'+ case.split('.')[0] + "_gt.png", (label.transpose(1, 2, 0)*255).astype(np.uint8))
    return metric_list
