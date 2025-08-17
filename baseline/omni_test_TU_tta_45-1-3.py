import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from scipy import ndimage

from datasets.dataset_aug_norm_mc import CenterCropGenerator
from datasets.dataset_aug_norm_mc import USdatasetClsFlexible, USdatasetSegFlexible
from datasets.omni_dataset import position_prompt_one_hot_dict, task_prompt_one_hot_dict, type_prompt_one_hot_dict, nature_prompt_one_hot_dict

POSITION_LEN = len(position_prompt_one_hot_dict)
TASK_LEN = len(task_prompt_one_hot_dict)
TYPE_LEN = len(type_prompt_one_hot_dict)
NAT_LEN = len(nature_prompt_one_hot_dict)

from utils import omni_seg_test_TU, calculate_metric_percase
from sklearn.metrics import accuracy_score
import cv2  # 添加cv2导入，用于保存图像

# === 使用 TransUNet ===
from networks.vit_seg_modeling_v4 import VisionTransformer, CONFIGS as VIT_CONFIGS


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='data/', help='root dir for data')
parser.add_argument('--output_dir', type=str, required=True, help='output dir')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per GPU for inference')
parser.add_argument('--img_size', type=int, default=224, help='input size (H=W)')
parser.add_argument('--is_saveout', action="store_true", help='save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction dir')
parser.add_argument('--deterministic', type=int, default=1, help='deterministic')
parser.add_argument('--seed', type=int, default=1234, help='random seed')

# —— TransUNet 相关 —— #
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                    choices=['ViT-B_16','ViT-B_32','ViT-L_16','ViT-L_32','ViT-H_14','R50-ViT-B_16','R50-ViT-L_16'])
parser.add_argument('--vit_pretrained', type=str, default=None,
                    help='.npz pretrained path (if None, use cfg.pretrained_path)')

# —— checkpoint —— #
parser.add_argument('--resume', type=str, required=True, help='model checkpoint (.pth) for inference')

parser.add_argument('--prompt', action='store_true', help='(deprecated here) using prompt for training')
parser.add_argument('--adapter_ft', action='store_true', help='(deprecated here) using adapter for fine-tuning')

# —— TTA 相关 —— #
parser.add_argument('--use_tta', action='store_true', help='enable test time augmentation (enabled for ALL datasets)')
parser.add_argument('--ms_scales_thyroid', type=str, default='0.9,1.0,1.1',
                    help='Multi-scale factors for TTA on private_Thyroid only, comma-separated.')
parser.add_argument('--ms_scales_default', type=str, default='1.0',
                    help='Scales for all other datasets (and all classification). Usually "1.0".')

args = parser.parse_args()


def _parse_scales(s):
    try:
        return [float(x) for x in str(s).split(',') if str(x).strip()]
    except Exception:
        return [1.0]

args.ms_scales_thyroid_list = _parse_scales(args.ms_scales_thyroid)
args.ms_scales_default_list = _parse_scales(args.ms_scales_default)


class TTAGenerator:
    """Test Time Augmentation Generator (with optional multi-scale)."""

    def __init__(self, img_size=224, scales=(1.0,)):
        self.img_size = img_size
        self.scales = tuple(scales)
        # ImageNet标准化参数
        self.IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.IMAGENET_DEFAULT_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def apply_geometric_transforms(self, image_np):
        """
        几何变换组（4个）：无、水平翻转、垂直翻转、水平+垂直翻转
        输入: numpy array (H, W, C), [0,1]
        """
        transforms = []
        transforms.append(image_np.copy())                           # 0: none
        transforms.append(np.flip(image_np, axis=1).copy())          # 1: hflip
        transforms.append(np.flip(image_np, axis=0).copy())          # 2: vflip
        temp = np.flip(image_np, axis=1)
        transforms.append(np.flip(temp, axis=0).copy())              # 3: hvflip
        return transforms

    def apply_photometric_transforms(self, image_np):
        """
        光度变换组（5个）：无、gamma(0.8)、gamma(1.2)、contrast(0.9)、contrast(1.1)
        输入: numpy array (H, W, C), [0,1]
        """
        transforms = []
        transforms.append(image_np.copy())                           # 0: none
        transforms.append(np.clip(np.power(image_np, 0.8), 0, 1))    # 1
        transforms.append(np.clip(np.power(image_np, 1.2), 0, 1))    # 2
        transforms.append(np.clip(image_np * 0.9, 0, 1))             # 3
        transforms.append(np.clip(image_np * 1.1, 0, 1))             # 4
        return transforms

    def normalize_and_convert(self, image_np):
        """
        标准化并转换为tensor (HWC)，内部会做 CHW 标准化再转回 HWC，方便复用你原有 to_chw()
        """
        image_tensor = torch.from_numpy(image_np.astype(np.float32)).permute(2, 0, 1)
        image_tensor = (image_tensor - self.IMAGENET_DEFAULT_MEAN) / self.IMAGENET_DEFAULT_STD
        image_tensor = image_tensor.permute(1, 2, 0)  # CHW -> HWC
        return image_tensor

    def _scale_and_fit(self, image_np, scale):
        """
        多尺度：先按比例缩放到 new_size，再中心裁剪/反射填充回固定输入尺寸，不改变模型输入大小。
        """
        import cv2
        target = self.img_size
        new_size = max(1, int(round(target * float(scale))))
        scaled = cv2.resize(image_np.astype(np.float32), (new_size, new_size), interpolation=cv2.INTER_LINEAR)

        if new_size > target:
            s = (new_size - target) // 2
            scaled = scaled[s:s+target, s:s+target, :]
        elif new_size < target:
            pad = target - new_size
            top = pad // 2
            bottom = pad - top
            left = pad // 2
            right = pad - left
            scaled = np.pad(scaled, ((top, bottom), (left, right), (0, 0)), mode='reflect')

        if scaled.shape[0] != target or scaled.shape[1] != target:
            scaled = cv2.resize(scaled, (target, target), interpolation=cv2.INTER_LINEAR)

        return np.clip(scaled, 0, 1).astype(np.float32)

    def generate_tta_samples(self, sample):
        """
        生成 TTA 样本：
        - 所有数据集：几何×光度
        - 多尺度：由 self.scales 控制（Thyroid 多尺度；其他数据集通常为 [1.0]）
        """
        image = sample['image']  # HWC tensor 或 numpy
        nature = sample.get('nature_for_aug', 'unknown')

        # 逆标准化回 [0,1] numpy
        if isinstance(image, torch.Tensor):
            image_chw = image.permute(2, 0, 1)
            image_chw = image_chw * self.IMAGENET_DEFAULT_STD + self.IMAGENET_DEFAULT_MEAN
            image_np = image_chw.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image
        image_np = np.clip(image_np, 0, 1).astype(np.float32)

        tta_samples = []
        for s in self.scales:
            base_np = self._scale_and_fit(image_np, s)

            if nature == "tumor":
                # tumor: 4 geo × 5 photo
                geo_transforms = self.apply_geometric_transforms(base_np)
                for geo_idx, geo_img in enumerate(geo_transforms):
                    for pid in range(5):
                        if pid == 0:
                            final_img = geo_img.copy()
                        elif pid == 1:
                            final_img = np.clip(np.power(geo_img, 0.8), 0, 1)
                        elif pid == 2:
                            final_img = np.clip(np.power(geo_img, 1.2), 0, 1)
                        elif pid == 3:
                            final_img = np.clip(geo_img * 0.9, 0, 1)
                        else:
                            final_img = np.clip(geo_img * 1.1, 0, 1)
                        tta_sample = sample.copy()
                        tta_sample['image'] = self.normalize_and_convert(final_img)
                        tta_sample['geo_idx'] = geo_idx
                        tta_samples.append(tta_sample)
            else:
                # 非 tumor: 仅 5 个光度（无需记录翻转）
                photo_transforms = self.apply_photometric_transforms(base_np)
                for photo_img in photo_transforms:
                    tta_sample = sample.copy()
                    tta_sample['image'] = self.normalize_and_convert(photo_img)
                    tta_sample['geo_idx'] = 0
                    tta_samples.append(tta_sample)

        return tta_samples

    # ==== 融合函数（保持 logits 空间融合 + 仅逆翻转）====
    def merge_predictions_seg(self, predictions_list, geo_indices):
        """
        合并分割预测结果（logits 空间）：
        1) 按 geo_indices 做几何逆变换（仅翻转）到原图坐标；
        2) 对每个视图的 logits 求平均；
        3) 外部再 softmax/argmax。
        """
        if len(predictions_list) == 0:
            return None
        corrected_logits = []
        for logit, geo_idx in zip(predictions_list, geo_indices):
            if geo_idx == 0:
                corrected_logits.append(logit)
            elif geo_idx == 1:  # 逆水平翻转
                corrected_logits.append(torch.flip(logit, dims=[2]))
            elif geo_idx == 2:  # 逆垂直翻转
                corrected_logits.append(torch.flip(logit, dims=[1]))
            elif geo_idx == 3:  # 逆水平+垂直翻转
                tmp = torch.flip(logit, dims=[2])
                corrected_logits.append(torch.flip(tmp, dims=[1]))
            else:
                corrected_logits.append(logit)
        merged_logit = torch.stack(corrected_logits, dim=0).mean(dim=0)  # (C, H, W)
        return merged_logit

    def merge_predictions_seg_Probability(self, predictions_list, geo_indices):
        """
        合并分割预测（概率图求平均）
        """
        if len(predictions_list) == 0:
            return None
        corrected_preds = []
        for pred, geo_idx in zip(predictions_list, geo_indices):
            if geo_idx == 0:
                corrected_preds.append(pred)
            elif geo_idx == 1:
                corrected_preds.append(torch.flip(pred, dims=[2]))
            elif geo_idx == 2:
                corrected_preds.append(torch.flip(pred, dims=[1]))
            elif geo_idx == 3:
                temp = torch.flip(pred, dims=[2])
                corrected_preds.append(torch.flip(temp, dims=[1]))
        merged_pred = torch.stack(corrected_preds, dim=0).mean(dim=0)
        return merged_pred

    def merge_predictions_cls(self, predictions_list):
        """
        合并分类预测结果 (logits 均值)
        """
        if len(predictions_list) == 0:
            return None
        return torch.stack(predictions_list, dim=0).mean(dim=0)


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_ddp():
    # torchrun 会自动设置这些环境变量
    if 'LOCAL_RANK' not in os.environ:
        # 允许单进程无 torchrun 运行
        return None, torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda', local_rank)
    return local_rank, device


def omni_seg_test_TU_with_tta(image, label, merged_prediction, classes, ClassStartIndex=1,
                              test_save_path=None, case=None, dataset_name=None):
    """
    使用预计算的TTA预测结果进行分割评估
    merged_prediction: 已经合并的TTA预测结果 (C, H, W) tensor
    """
    label = label.cpu().detach().numpy()
    image_save = image.cpu().detach().numpy()

    # 使用预计算的合并预测
    seg_out = merged_prediction.unsqueeze(0)  # (1, C, H, W)
    out_label_back_transform = torch.cat(
        [seg_out[:, 0:1], seg_out[:, ClassStartIndex:ClassStartIndex+classes-1]], axis=1)
    out = torch.argmax(torch.softmax(out_label_back_transform, dim=1), dim=1)
    prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        image = (image_save - np.min(image_save)) / (np.max(image_save) - np.min(image_save))
        os.makedirs(test_save_path + '/' + dataset_name, exist_ok=True)
        cv2.imwrite(test_save_path + '/'+ dataset_name + '/'+case + "_pred.png", (prediction.transpose(1, 2, 0)*255).astype(np.uint8))
        cv2.imwrite(test_save_path + '/'+ dataset_name + '/'+case + "_img.png", ((image.squeeze(0))*255).astype(np.uint8))
        cv2.imwrite(test_save_path + '/'+ dataset_name + '/'+case + "_gt.png", (label.transpose(1, 2, 0)*255).astype(np.uint8))

    return metric_list


# NHWC -> NCHW（如果需要）
def to_chw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous() if x.dim() == 4 and x.size(-1) in (1, 3) else x


@torch.no_grad()
def inference(args, model, device, test_save_path=None):
    import csv, time

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        result_csv = f"{args.output_dir}/tta_result_{os.path.splitext(os.path.basename(args.resume))[0]}.csv"
        if not os.path.exists(result_csv):
            with open(result_csv, 'w', newline='') as f:
                csv.writer(f).writerow(['dataset', 'task', 'metric', 'time'])
    else:
        result_csv = None

    seg_test_set = [
        "private_Thyroid",
        "private_Kidney",
        "private_Fetal_Head",
        "private_Cardiac",
        "private_Breast_luminal",
        "private_Breast",
    ]
    cls_test_set = [
        "private_Liver",
        "private_Breast_luminal",
        "private_Breast",
        "private_Appendix",
    ]

    private_performence_seg = {}
    private_performence_cls = {}

    # ===== Segmentation =====
    for dataset_name in seg_test_set:
        num_classes = 2
        ds = USdatasetSegFlexible(
            base_dir=os.path.join(args.root_path, "segmentation", dataset_name),
            split="test",
            list_dir=os.path.join(args.root_path, "segmentation", dataset_name),
            transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt
        )
        sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if dist.is_initialized() else None
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False if sampler else False,
                            num_workers=12, pin_memory=True, sampler=sampler)

        if is_main_process():
            logging.info("%s: %d samples", dataset_name, len(ds))

        # —— 为当前数据集选择尺度（Thyroid 多尺度，其余 1.0）——
        if args.use_tta:
            cur_scales = args.ms_scales_thyroid_list if dataset_name == "private_Thyroid" \
                         else args.ms_scales_default_list
            tta_generator_cur = TTAGenerator(img_size=args.img_size, scales=cur_scales)
        else:
            tta_generator_cur = None

        model.eval()
        metric_list = 0.0
        count_matrix = np.ones((len(ds), num_classes - 1))
        iterator = tqdm(enumerate(loader), total=len(loader), disable=not is_main_process())

        base_idx = 0
        for i_batch, sampled_batch in iterator:
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name']
            case_name = case_name[0] if isinstance(case_name, (list, tuple)) else str(case_name)

            if args.use_tta:
                # 注意：batch_size=1
                if image.size(0) != 1:
                    raise ValueError("TTA mode requires batch_size=1")

                # 准备单个sample
                single_sample = {
                    'image': image[0],
                    'label': label[0] if label is not None else None,
                    'nature_for_aug': sampled_batch.get('nature_for_aug', ['unknown'])[0]
                }
                if args.prompt:
                    single_sample['position_prompt'] = sampled_batch['position_prompt']
                    single_sample['type_prompt']     = sampled_batch['type_prompt']
                    single_sample['nature_prompt']   = sampled_batch['nature_prompt']

                # 生成 TTA 样本（Thyroid 会是多尺度，其余为单尺度）
                tta_samples = tta_generator_cur.generate_tta_samples(single_sample)

                # 逐个前向
                tta_predictions = []
                geo_indices = []
                for tta_sample in tta_samples:
                    tta_image = to_chw(tta_sample['image'].unsqueeze(0)).to(device, non_blocking=True)
                    geo_indices.append(tta_sample['geo_idx'])

                    if args.prompt:
                        position_prompt = torch.tensor(np.array(tta_sample['position_prompt'])).permute([1, 0]).float()
                        task_prompt     = torch.tensor(np.array([[1], [0]])).permute([1, 0]).float()
                        type_prompt     = torch.tensor(np.array(tta_sample['type_prompt'])).permute([1, 0]).float()
                        nature_prompt   = torch.tensor(np.array(tta_sample['nature_prompt'])).permute([1, 0]).float()

                        outputs = model(tta_image, position_prompt.cuda(), task_prompt.cuda(),
                                        type_prompt.cuda(), nature_prompt.cuda())
                    else:
                        outputs = model(tta_image)

                    seg_logit = outputs[0]               # (1, C, H, W), logits
                    tta_predictions.append(seg_logit[0]) # (C, H, W)

                # 融合
                merged_pred = tta_generator_cur.merge_predictions_seg(tta_predictions, geo_indices)

                # 评估
                metric_i = omni_seg_test_TU_with_tta(
                    image.to(device), label, merged_pred,
                    classes=num_classes,
                    test_save_path=test_save_path,
                    case=case_name,
                    dataset_name=dataset_name
                )

            else:
                # 原始模式
                image = to_chw(image).to(device, non_blocking=True)
                if args.prompt:
                    position_prompt = torch.tensor(np.array(sampled_batch['position_prompt'])).permute([1, 0]).float()
                    task_prompt     = torch.tensor(np.array([[1], [0]])).permute([1, 0]).float()
                    type_prompt     = torch.tensor(np.array(sampled_batch['type_prompt'])).permute([1, 0]).float()
                    nature_prompt   = torch.tensor(np.array(sampled_batch['nature_prompt'])).permute([1, 0]).float()
                    metric_i = omni_seg_test_TU(image, label, model,
                                             classes=num_classes,
                                             test_save_path=test_save_path,
                                             case=case_name,
                                             prompt=args.prompt,
                                             type_prompt=type_prompt,
                                             nature_prompt=nature_prompt,
                                             position_prompt=position_prompt,
                                             task_prompt=task_prompt,
                                             dataset_name=dataset_name
                                             )
                else:
                    metric_i = omni_seg_test_TU(image, label, model,
                                             classes=num_classes,
                                             test_save_path=test_save_path,
                                             case=case_name, dataset_name=dataset_name)

            zero_label_flag = False
            for i in range(1, num_classes):
                if not metric_i[i - 1][1]:
                    idx = base_idx + i_batch
                    if idx < len(ds):
                        count_matrix[idx, i - 1] = 0
                    zero_label_flag = True

            metric_vals = metric_i[0][0]
            metric_list += np.array(metric_vals).sum()

            if is_main_process():
                iterator.set_postfix(mean_dice=float(np.mean(metric_vals)))

        # 同步 reduce
        if dist.is_initialized():
            tensor_ml = torch.tensor(metric_list, device=device)
            tensor_cm = torch.tensor(count_matrix, device=device)
            dist.all_reduce(tensor_ml, op=dist.ReduceOp.SUM)
            dist.all_reduce(tensor_cm, op=dist.ReduceOp.SUM)
            metric_list = tensor_ml.item()
            count_matrix = tensor_cm.cpu().numpy()

        metric_list = metric_list / (count_matrix.sum(axis=0) + 1e-6)
        performance = float(np.mean(metric_list, axis=0))

        if is_main_process():
            task_name = 'transunet_seg_tta' if args.use_tta else 'transunet_seg_ddp'
            logging.info('[SEG] %s mean_dice: %.4f %s', dataset_name, performance, '(TTA)' if args.use_tta else '')
            if "private_" in dataset_name:
                private_performence_seg[dataset_name] = performance
            with open(result_csv, 'a', newline='') as f:
                csv.writer(f).writerow([dataset_name, task_name, performance,
                                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])

    # ===== Classification =====
    for dataset_name in cls_test_set:
        num_classes = 4 if dataset_name == "private_Breast_luminal" else 2

        ds = USdatasetClsFlexible(
            base_dir=os.path.join(args.root_path, "classification", dataset_name),
            split="test",
            list_dir=os.path.join(args.root_path, "classification", dataset_name),
            transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt
        )
        sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if dist.is_initialized() else None
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False if sampler else False,
                            num_workers=2, pin_memory=True, sampler=sampler)

        if is_main_process():
            logging.info("%s: %d samples", dataset_name, len(ds))

        # 分类：也启用 TTA，但全部使用 default 尺度（通常 1.0）
        if args.use_tta:
            tta_generator_cur = TTAGenerator(img_size=args.img_size, scales=args.ms_scales_default_list)
        else:
            tta_generator_cur = None

        model.eval()
        label_list = []
        prediction_list = []
        iterator = tqdm(enumerate(loader), total=len(loader), disable=not is_main_process())

        for i_batch, sampled_batch in iterator:
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name']
            case_name = case_name[0] if isinstance(case_name, (list, tuple)) else str(case_name)

            if args.use_tta:
                if image.size(0) != 1:
                    raise ValueError("TTA mode requires batch_size=1")

                single_sample = {
                    'image': image[0],
                    'label': label[0] if label is not None else None,
                    'nature_for_aug': sampled_batch.get('nature_for_aug', ['unknown'])[0]
                }
                if args.prompt:
                    single_sample['position_prompt'] = sampled_batch['position_prompt']
                    single_sample['type_prompt']     = sampled_batch['type_prompt']
                    single_sample['nature_prompt']   = sampled_batch['nature_prompt']

                tta_samples = tta_generator_cur.generate_tta_samples(single_sample)

                tta_logits = []
                for tta_sample in tta_samples:
                    tta_image = to_chw(tta_sample['image'].unsqueeze(0)).to(device, non_blocking=True)

                    if args.prompt:
                        position_prompt = torch.tensor(np.array(tta_sample['position_prompt'])).permute([1, 0]).float()
                        task_prompt     = torch.tensor(np.array([[0], [1]])).permute([1, 0]).float()
                        type_prompt     = torch.tensor(np.array(tta_sample['type_prompt'])).permute([1, 0]).float()
                        nature_prompt   = torch.tensor(np.array(tta_sample['nature_prompt'])).permute([1, 0]).float()

                        outputs = model(tta_image, position_prompt.cuda(), task_prompt.cuda(),
                                        type_prompt.cuda(), nature_prompt.cuda())
                    else:
                        outputs = model(tta_image)

                    # 获取分类logits
                    logits = outputs[2] if num_classes == 4 else outputs[1]
                    tta_logits.append(logits[0])  # (C,)

                # 合并 TTA 预测
                merged_logits = tta_generator_cur.merge_predictions_cls(tta_logits)
                preds = torch.argmax(torch.softmax(merged_logits.unsqueeze(0), dim=1), dim=1)

            else:
                # 原始模式
                image = to_chw(image).to(device, non_blocking=True)

                if args.prompt:
                    position_prompt = torch.tensor(np.array(sampled_batch['position_prompt'])).permute([1, 0]).float()
                    task_prompt     = torch.tensor(np.array([[0], [1]])).permute([1, 0]).float()
                    type_prompt     = torch.tensor(np.array(sampled_batch['type_prompt'])).permute([1, 0]).float()
                    nature_prompt   = torch.tensor(np.array(sampled_batch['nature_prompt'])).permute([1, 0]).float()
                    outputs = model(image, position_prompt.cuda(), task_prompt.cuda(),
                                    type_prompt.cuda(), nature_prompt.cuda())
                else:
                    outputs = model(image)

                logits = outputs[2] if num_classes == 4 else outputs[1]
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            label_list.append(label.detach().cpu())
            prediction_list.append(preds.detach().cpu())

        # gather & concat across ranks
        labels = torch.cat(label_list, dim=0) if len(label_list) else torch.empty(0, dtype=torch.long)
        preds  = torch.cat(prediction_list, dim=0) if len(prediction_list) else torch.empty(0, dtype=torch.long)

        if dist.is_initialized():
            labels = labels.to(device, non_blocking=True)
            preds  = preds.to(device, non_blocking=True)

            gather_labels = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
            gather_preds  = [torch.empty_like(preds)  for _ in range(dist.get_world_size())]
            dist.all_gather(gather_labels, labels)
            dist.all_gather(gather_preds,  preds)

            labels = torch.cat(gather_labels, dim=0)
            preds  = torch.cat(gather_preds,  dim=0)

        labels = labels.detach().cpu().numpy()
        preds  = preds.detach().cpu().numpy()
        print("labels: ", labels)
        print("preds: ", preds)

        if labels.size > 0:
            performance = accuracy_score(labels, preds)
        else:
            performance = 0.0

        if is_main_process():
            task_name = 'transunet_cls_tta' if args.use_tta else 'transunet_cls_ddp'
            logging.info('[CLS] %s acc: %.4f %s', dataset_name, performance, '(TTA)' if args.use_tta else '')
            if "private_" in dataset_name:
                private_performence_cls[dataset_name] = performance
            with open(result_csv, 'a', newline='') as f:
                csv.writer(f).writerow([dataset_name, task_name, performance,
                                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])

    # ===== Summary =====
    if is_main_process():
        if private_performence_seg:
            logging.info("Private segmentation performance:")
            for k, v in private_performence_seg.items():
                logging.info("%s: %.4f", k, v)
            logging.info("Mean Dice: %.4f", np.mean(list(private_performence_seg.values())))
        if private_performence_cls:
            logging.info("Private classification performance:")
            for k, v in private_performence_cls.items():
                logging.info("%s: %.4f", k, v)
            logging.info("Mean Acc: %.4f", np.mean(list(private_performence_cls.values())))


def build_transunet_and_load(args, device):
    # 1) 配置
    cfg = VIT_CONFIGS[args.vit_name]
    if hasattr(cfg, "patches") and getattr(cfg.patches, "grid", None) is not None:
        if args.img_size == 224:
            try:
                if isinstance(cfg.patches, dict):
                    gh, gw = cfg.patches["grid"]
                else:
                    gh, gw = cfg.patches.grid
                if (gh, gw) != (14, 14):
                    if isinstance(cfg.patches, dict):
                        cfg.patches["grid"] = (14, 14)
                    else:
                        cfg.patches.grid = (14, 14)
            except Exception:
                pass
    cfg.n_classes = 2

    # 2) 模型
    net = VisionTransformer(
        config=cfg,
        img_size=args.img_size,
        num_classes=cfg.n_classes,
        zero_head=False,
        vis=False,
        prompt=args.prompt,
        pos_len=POSITION_LEN,
        task_len=TASK_LEN,
        type_len=TYPE_LEN,
        nature_len=NAT_LEN
    ).cuda()

    # 3) npz 预训练
    npz_path = args.vit_pretrained or getattr(cfg, "pretrained_path", None)
    if npz_path and os.path.isfile(npz_path):
        if is_main_process():
            print(f"[TransUNet] Loading pretrained (npz) from {npz_path}")
        weights = np.load(npz_path, allow_pickle=True)
        net.load_from(weights)

    # 4) 本地 checkpoint（.pth）
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Resume checkpoint file not found: {args.resume}")
    if is_main_process():
        print("Loading checkpoint from ", args.resume)
    payload = torch.load(args.resume, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict) else payload

    new_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_state[nk] = v
    missing, unexpected = net.load_state_dict(new_state, strict=False)
    if is_main_process():
        print(f"[TransUNet] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    return net


if __name__ == "__main__":
    # ===== 随机数 & CUDNN =====
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ===== TTA 检查 =====
    if args.use_tta and args.batch_size != 1:
        if is_main_process():
            print(f"Warning: TTA mode requires batch_size=1, but got {args.batch_size}. Setting batch_size=1.")
        args.batch_size = 1

    # ===== DDP =====
    local_rank, device = setup_ddp()

    # ===== 模型 & DDP 包装 =====
    net = build_transunet_and_load(args, device)
    if dist.is_initialized():
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    # ===== 日志（仅 rank0 写文件）=====
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, f"tta_test_result_{os.path.splitext(os.path.basename(args.resume))[0]}.txt")
        with open(log_path, "a"):
            pass
        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(os.path.basename(args.resume))
        if args.use_tta:
            logging.info("TTA enabled for ALL datasets. Multi-scale ONLY for private_Thyroid.")
            logging.info("Scales: Thyroid=%s ; Others(Class&Seg)=%s",
                         args.ms_scales_thyroid, args.ms_scales_default)

    # ===== 保存输出目录 =====
    if args.is_saveout:
        if args.use_tta:
            args.test_save_dir = os.path.join(args.output_dir, "predictions_tta")
        else:
            args.test_save_dir = os.path.join(args.output_dir, "predictions")
        if is_main_process():
            os.makedirs(args.test_save_dir, exist_ok=True)
        test_save_path = args.test_save_dir
    else:
        test_save_path = None

    # 同步一下目录创建
    if dist.is_initialized():
        dist.barrier()

    # ===== 推理 =====
    inference(args, net, device, test_save_path)

    # 关闭进程组
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
