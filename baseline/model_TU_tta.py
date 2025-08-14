# sample_result_submission/model.py

import os
import cv2
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F

from config_PG import get_config
from networks.vit_seg_modeling_v2 import VisionTransformer, CONFIGS as VIT_CONFIGS
from datasets.dataset_aug_norm_mc import CenterCropGenerator

# -------- Prompt 相关映射 --------
organ_to_position_map = {
    'Breast': 'breast', 'Heart': 'cardiac', 'Thyroid': 'thyroid', 'Head': 'head',
    'Kidney': 'kidney', 'Appendix': 'appendix', 'Liver': 'liver'
}
position_prompt_one_hot_dict = {
    "breast":[1,0,0,0,0,0,0,0], "cardiac":[0,1,0,0,0,0,0,0], "thyroid":[0,0,1,0,0,0,0,0],
    "head":[0,0,0,1,0,0,0,0], "kidney":[0,0,0,0,1,0,0,0], "appendix":[0,0,0,0,0,1,0,0],
    "liver":[0,0,0,0,0,0,1,0], "indis":[0,0,0,0,0,0,0,1]
}
task_prompt_one_hot_dict = {"segmentation":[1,0],"classification":[0,1]}
organ_to_nature_map = {
    'Breast':'tumor','Heart':'organ','Thyroid':'tumor','Head':'organ',
    'Kidney':'organ','Appendix':'organ','Liver':'organ'
}
nature_prompt_one_hot_dict = {"tumor":[1,0],"organ":[0,1]}
type_prompt_one_hot_dict = {"whole":[1,0,0],"local":[0,1,0],"location":[0,0,1]}


class TTAGenerator:
    """Test Time Augmentation Generator"""
    
    def __init__(self, img_size=224, transform=None):
        self.img_size = img_size
        self.transform = transform
        # ImageNet标准化参数
        self.IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def apply_geometric_transforms(self, image_np):
        """
        几何变换组 (4个)
        输入: numpy array (H, W, C), 值域[0,1]
        输出: list of numpy arrays
        """
        transforms = []
        
        # 1. 原图
        transforms.append(image_np.copy())
        
        # 2. 水平翻转
        transforms.append(np.flip(image_np, axis=1).copy())
        
        # 3. 垂直翻转
        transforms.append(np.flip(image_np, axis=0).copy())
        
        # 4. 水平+垂直翻转
        temp = np.flip(image_np, axis=1)
        transforms.append(np.flip(temp, axis=0).copy())
        
        return transforms
    
    def apply_photometric_transforms(self, image_np):
        """
        光度变换组 (5个)
        输入: numpy array (H, W, C), 值域[0,1]
        输出: list of numpy arrays
        """
        transforms = []
        
        # 1. 原图
        transforms.append(image_np.copy())
        
        # 2. Gamma=0.8
        img_gamma_08 = np.power(image_np, 0.8)
        transforms.append(np.clip(img_gamma_08, 0, 1))
        
        # 3. Gamma=1.2
        img_gamma_12 = np.power(image_np, 1.2)
        transforms.append(np.clip(img_gamma_12, 0, 1))
        
        # 4. 对比度=0.9
        img_contrast_09 = image_np * 0.9
        transforms.append(np.clip(img_contrast_09, 0, 1))
        
        # 5. 对比度=1.1
        img_contrast_11 = image_np * 1.1
        transforms.append(np.clip(img_contrast_11, 0, 1))
        
        return transforms
    
    def generate_tta_samples(self, image_np, nature='organ'):
        """
        生成所有TTA样本 - 先应用transform预处理，再应用TTA变换
        - tumor: 4个几何变换 × 5个光度变换 = 20个变换
        - 非tumor: 只用5个光度变换
        返回: list of {'image': tensor, 'geo_idx': int}
        """

        # 先使用transform进行预处理 (resize + crop)
        sample = {'image': image_np, 'label': np.zeros(image_np.shape[:2])}
        if self.transform:
            processed_sample = self.transform(sample)

            # 逆标准化回到[0,1]范围的numpy array
            if isinstance(processed_sample['image'], torch.Tensor):
                # HWC -> CHW
                image_chw = processed_sample['image'].permute(2, 0, 1)
                # 逆标准化
                image_chw = image_chw * self.IMAGENET_DEFAULT_STD + self.IMAGENET_DEFAULT_MEAN
                # CHW -> HWC, 转为numpy
                processed_image_np = image_chw.permute(1, 2, 0).numpy()
            else:
                processed_image_np = processed_sample['image']

        else:
            processed_image_np = image_np
        
        # 确保值域在[0,1]
        processed_image_np = np.clip(processed_image_np, 0, 1)
        
        tta_samples = []
        
        if nature == "tumor":
            # tumor: 使用几何变换 × 光度变换的完全组合
            geo_transforms = self.apply_geometric_transforms(processed_image_np)  # 4个
            
            # 完全组合：每个几何变换 × 每个光度变换
            for geo_idx, geo_img in enumerate(geo_transforms):
                photo_transforms = self.apply_photometric_transforms(geo_img)  # 5个
                
                for photo_idx, photo_img in enumerate(photo_transforms):
                    # 转换为tensor并标准化: (H, W, C) -> (C, H, W)
                    image_tensor = torch.from_numpy(photo_img.astype(np.float32)).permute(2, 0, 1)
                    image_tensor = (image_tensor - self.IMAGENET_DEFAULT_MEAN) / self.IMAGENET_DEFAULT_STD
                    
                    tta_sample = {
                        'image': image_tensor,
                        'geo_idx': geo_idx  # 记录几何变换索引，用于逆变换
                    }
                    tta_samples.append(tta_sample)
        else:
            # 非tumor: 只使用光度变换（几何变换索引都是0，即原图）
            photo_transforms = self.apply_photometric_transforms(processed_image_np)  # 5个
            
            for photo_idx, photo_img in enumerate(photo_transforms):
                image_tensor = torch.from_numpy(photo_img.astype(np.float32)).permute(2, 0, 1)
                image_tensor = (image_tensor - self.IMAGENET_DEFAULT_MEAN) / self.IMAGENET_DEFAULT_STD
                
                tta_sample = {
                    'image': image_tensor,
                    'geo_idx': 0  # 非tumor只用原图几何形状
                }
                tta_samples.append(tta_sample)
        
        return tta_samples
    
    def merge_predictions_seg(self, predictions_list, geo_indices):
        """
        合并分割预测结果 (对概率图求平均)
        predictions_list: list of tensors, each shape (C, H, W)
        geo_indices: list of geometric transform indices for inverse transformation
        """
        if len(predictions_list) == 0:
            return None
        
        # 根据几何变换索引进行逆变换
        corrected_preds = []
        
        for pred, geo_idx in zip(predictions_list, geo_indices):
            if geo_idx == 0:
                # 原图 - 不变
                corrected_preds.append(pred)
            elif geo_idx == 1:
                # 水平翻转 - 逆水平翻转
                corrected_preds.append(torch.flip(pred, dims=[2]))
            elif geo_idx == 2:
                # 垂直翻转 - 逆垂直翻转
                corrected_preds.append(torch.flip(pred, dims=[1]))
            elif geo_idx == 3:
                # 水平+垂直翻转 - 逆水平+垂直翻转
                temp = torch.flip(pred, dims=[2])  # 逆水平翻转
                corrected_preds.append(torch.flip(temp, dims=[1]))  # 逆垂直翻转
        
        # 求平均
        merged_pred = torch.stack(corrected_preds, dim=0).mean(dim=0)
        return merged_pred
    
    def merge_predictions_cls(self, predictions_list):
        """
        合并分类预测结果 (对logits求平均)
        predictions_list: list of tensors, each shape (num_classes,)
        """
        if len(predictions_list) == 0:
            return None
        
        # 直接对所有logits求平均 (分类任务不需要逆变换)
        merged_pred = torch.stack(predictions_list, dim=0).mean(dim=0)
        return merged_pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    p.add_argument('--data_list', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--checkpoint', required=True, help='model ckpt (.pth)')
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--use_prompts', action='store_true', default=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--vit_type', type=str, default='R50-ViT-B_16',
                   choices=VIT_CONFIGS.keys(),
                   help='Vision Transformer type, e.g., R50-ViT-B_16')
    # 添加TTA相关参数
    p.add_argument('--use_tta', action='store_true', help='enable test time augmentation')
    return p.parse_args()

def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model', ckpt)

    # 去掉或添加 module. 以匹配
    def try_load(sd):
        missing, unexpected = model.load_state_dict(sd, strict=False)
        return missing, unexpected
    
    def strip_module(sd):
        return { (k[7:] if k.startswith('module.') else k): v for k,v in sd.items() }

    try:
        m,u = try_load(strip_module(state))
        print(f'Checkpoint loaded (missing {len(m)}, unexpected {len(u)})')
    except Exception as e:
        print(f'Load attempt failed: {e}')
        
    return ckpt.get('epoch', 0)

def to_chw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous() if x.dim() == 4 and x.size(-1) in (1, 3) else x

class InferenceModel:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        class CfgArgs:
            img_size = args.img_size
            prompt = args.use_prompts
            # 其余训练无关占位
            opts=None; batch_size=None; zip=False; cache_mode=None; resume=None
            accumulation_steps=None; use_checkpoint=False; amp_opt_level=''
            tag=None; eval=True; throughput=False

        cfg_args = CfgArgs()
        config = VIT_CONFIGS[args.vit_type]
        self.net = VisionTransformer(config, prompt=cfg_args.prompt).to(self.device)
        load_checkpoint(self.net, args.checkpoint, self.device)
        self.net.eval()
        self.transform = CenterCropGenerator(output_size=[args.img_size, args.img_size])
        
        # 初始化TTA生成器 - 传入transform
        self.tta_generator = TTAGenerator(img_size=args.img_size, transform=self.transform) if args.use_tta else None
        
        if args.use_tta:
            print(f"TTA enabled: tumor (4 geo × 5 photo = 20), non-tumor (5 photo only)")

    def _build_prompts(self, task, organ):
        # task
        task_vec = task_prompt_one_hot_dict[task]
        task_prompt = torch.tensor(task_vec, dtype=torch.float).unsqueeze(0).to(self.device)
        # position
        pos_key = organ_to_position_map.get(organ, 'indis')
        pos_vec = position_prompt_one_hot_dict[pos_key]
        position_prompt = torch.tensor(pos_vec, dtype=torch.float).unsqueeze(0).to(self.device)
        # nature
        nat_key = organ_to_nature_map.get(organ, 'organ')
        nat_vec = nature_prompt_one_hot_dict[nat_key]
        nature_prompt = torch.tensor(nat_vec, dtype=torch.float).unsqueeze(0).to(self.device)
        # type (固定 whole，可按需逻辑扩展)
        type_vec = type_prompt_one_hot_dict['whole']
        type_prompt = torch.tensor(type_vec, dtype=torch.float).unsqueeze(0).to(self.device)
        return position_prompt, task_prompt, type_prompt, nature_prompt

    def _infer_forward(self, image_tensor, task, organ):
        if self.args.use_prompts:
            pos_p, task_p, type_p, nat_p = self._build_prompts(task, organ)
            outputs = self.net(
                image_tensor,
                pos_p, task_p, type_p, nat_p
            )
        else:
            outputs = self.net(image_tensor)
        return outputs

    def _predict_with_tta(self, img_np, task, organ, orig_size, dataset_name):
        """使用TTA进行预测"""
        # 获取nature信息用于TTA策略选择
        nature = organ_to_nature_map.get(organ, 'organ')
        
        # 生成TTA样本 (已经包含了transform预处理)
        tta_samples = self.tta_generator.generate_tta_samples(img_np, nature)
        
        if task == 'segmentation':
            # 分割任务TTA
            tta_predictions = []
            geo_indices = []
            
            for tta_sample in tta_samples:
                tta_image = tta_sample['image'].unsqueeze(0).to(self.device)  # [1, C, H, W]
                geo_indices.append(tta_sample['geo_idx'])
                
                with torch.no_grad():
                    outputs = self._infer_forward(tta_image, task, organ)
                    
                    if isinstance(outputs, dict):
                        seg_logits = outputs.get('seg_logits')
                    else:
                        seg_logits = outputs[0]
                    
                    # 获取概率图
                    seg_probs = torch.softmax(seg_logits, dim=1)  # (1, C, H, W)
                    tta_predictions.append(seg_probs[0])  # 移除batch维度: (C, H, W)
            
            # 合并TTA预测
            merged_probs = self.tta_generator.merge_predictions_seg(tta_predictions, geo_indices)
            
            # 生成最终预测
            seg_pred = torch.argmax(merged_probs, dim=0)  # (H, W)
            mask_small = seg_pred.cpu().numpy().astype(np.uint8)
            
            # 二值化并缩放
            if mask_small.max() <= 1:
                mask_small = (mask_small * 255).astype(np.uint8)
            resized = cv2.resize(mask_small, orig_size, interpolation=cv2.INTER_NEAREST)
            
            return resized
            
        else:  # classification
            # 分类任务TTA
            tta_logits = []
            
            for tta_sample in tta_samples:
                tta_image = tta_sample['image'].unsqueeze(0).to(self.device)  # [1, C, H, W]
                
                with torch.no_grad():
                    outputs = self._infer_forward(tta_image, task, organ)
                    
                    if isinstance(outputs, dict):
                        logits = outputs.get('cls_logits')
                    else:
                        # 根据数据集选择logits
                        if dataset_name == 'Breast_luminal':
                            logits = outputs[2]  # 4类
                            num_classes = 4
                        else:
                            logits = outputs[1]  # 2类
                            num_classes = 2
                    
                    tta_logits.append(logits[0])  # 移除batch维度
            
            # 合并TTA预测
            merged_logits = self.tta_generator.merge_predictions_cls(tta_logits)
            
            # 生成最终预测
            probs = torch.softmax(merged_logits.unsqueeze(0), dim=1).cpu().numpy().squeeze(0)
            pred = int(probs.argmax())
            
            return {'probability': probs.tolist(), 'prediction': pred, 'num_classes': num_classes}

    def predict(self, data_list, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        cls_results = {}
        
        for item in tqdm(data_list, desc='Infer'):
            img_rel = item['img_path_relative']
            img_path = os.path.join(input_dir, img_rel)
            task = item['task']              # 'segmentation' or 'classification'
            dataset_name = item['dataset_name']
            organ = item['organ']

            img = Image.open(img_path).convert('RGB')
            orig_size = img.size  # (W,H)
            img_np = np.array(img) / 255.0  # 归一化到[0,1]
            
            if self.args.use_tta:
                # 使用TTA预测
                if task == 'segmentation':
                    resized_mask = self._predict_with_tta(img_np, task, organ, orig_size, dataset_name)
                    out_path = os.path.join(output_dir, img_rel.replace('img', 'mask'))
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    Image.fromarray(resized_mask).save(out_path)
                else:  # classification
                    result = self._predict_with_tta(img_np, task, organ, orig_size, dataset_name)
                    cls_results[img_rel] = result
            else:
                # 原始预测方式
                sample = {'image': img_np, 'label': np.zeros(img_np.shape[:2])}
                proc = self.transform(sample)
                image_tensor = proc['image'].unsqueeze(0).to(self.device)  # [1,C,H,W]
                image_tensor = to_chw(image_tensor)

                with torch.no_grad():
                    outputs = self._infer_forward(image_tensor, task, organ)

                if task == 'classification':
                    if isinstance(outputs, dict):
                        logits = outputs.get('cls_logits')
                    else:
                        if dataset_name.lower().find('luminal') != -1:
                            logits = outputs[2]  # 4 类
                            num_classes = 4
                        else:
                            logits = outputs[1]  # 2 类
                            num_classes = 2
                    probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze(0)
                    pred = int(probs.argmax())
                    cls_results[img_rel] = {'probability': probs.tolist(), 'prediction': pred, 'num_classes': num_classes}
                else:  # segmentation
                    if isinstance(outputs, dict):
                        seg_logits = outputs.get('seg_logits')
                    else:
                        seg_logits = outputs[0]
                    seg_pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1).squeeze(0)
                    mask_small = seg_pred.cpu().numpy().astype(np.uint8)
                    if mask_small.max() <= 1:
                        mask_small = (mask_small * 255).astype(np.uint8)
                    resized = cv2.resize(mask_small, orig_size, interpolation=cv2.INTER_NEAREST)
                    out_path = os.path.join(output_dir, img_rel.replace('img', 'mask'))
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    Image.fromarray(resized).save(out_path)

        # 保存分类结果
        if cls_results:
            with open(os.path.join(output_dir, 'classification.json'), 'w') as f:
                json.dump(cls_results, f, indent=2)
        
        tta_status = " (with TTA)" if self.args.use_tta else ""
        print(f'Done{tta_status}.')

def main():
    args = parse_args()
    with open(args.data_list, 'r') as f:
        data_list = json.load(f)
    model = InferenceModel(args)
    model.predict(data_list, args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()