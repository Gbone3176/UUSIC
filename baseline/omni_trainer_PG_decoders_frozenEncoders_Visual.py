import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from datasets.dataset_aug_norm import USdatasetSeg, USdatasetCls
from datasets.omni_dataset_decoders import USdatasetOmni_seg_decoders, USdatasetOmni_cls_decoders
from datasets.dataset_aug_norm import CenterCropGenerator, RandomGenerator_Seg, RandomGenerator_Cls

def save_tensor_as_image(tensor, save_path):
    """
    安全地保存张量为图像文件
    """
    # 确保张量在CPU上
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    
    # 处理不同的张量形状
    if len(tensor.shape) == 4:  # (B, C, H, W)
        tensor = tensor[0]  # 取第一个样本
    elif len(tensor.shape) == 3:  # (H, W, C)
        tensor = tensor.permute(2, 0, 1)  # 转换为 (C, H, W)
    elif len(tensor.shape) == 2:  # (H, W)
        tensor = tensor.unsqueeze(0)  # 添加通道维度
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
    
    # 确保数据类型正确
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    
    # 归一化到[0, 1]
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    if tensor_max > tensor_min:
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    
    # 如果是单通道图像，转换为RGB
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    
    # 保存图像
    save_image(tensor, save_path, normalize=False)

def visualize_input_images(args, snapshot_path):
    """
    可视化并保存输入模型的图像样本（不训练）
    功能：
    1. 遍历所有分割和分类数据集
    2. 保存原始图像和标签（分割mask或分类标签）
    3. 生成带标注的可视化结果
    """
    os.makedirs(os.path.join(snapshot_path, "visualization"), exist_ok=True)
    
    # ========== 分割数据集可视化 ==========
    seg_datasets = [
        'BUS-BRA', 
        'BUSI', 
        'BUSIS', 
        'CAMUS', 
        'DDTI', 
        'Fetal_HC', 
        'KidneyUS', 
        'private_Breast', 
        'private_Breast_luminal', 
        'private_Cardiac',
        'private_Fetal_Head', 
        'private_Kidney', 
        'private_Thyroid'
    ]
    
    for dataset_name in seg_datasets:
        dataset_path = os.path.join(args.root_path, "segmentation", dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Segmentation dataset not found: {dataset_path}")
            continue

        try:
            # 创建数据集和DataLoader
            db_seg = USdatasetOmni_seg_decoders(
                base_dir=dataset_path,
                split="train",  # 使用训练集可视化
                transform=RandomGenerator_Seg(output_size=[args.img_size, args.img_size]),
                prompt=args.prompt
            )
            
            dataloader = DataLoader(db_seg, batch_size=4, shuffle=True, num_workers=4)
            
            # 创建保存目录
            save_dir = os.path.join(snapshot_path, "visualization", "segmentation", dataset_name)
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"\nVisualizing segmentation dataset: {dataset_name}")
            print(f"Dataset size: {len(db_seg)}")
            
            for i, batch in enumerate(tqdm(dataloader, desc=f"Processing {dataset_name}")):
                if i >= 3:  # 每个数据集只可视化3个batch
                    break
                
                images = batch['image']
                masks = batch['label']
                
                print(f"Batch {i}: Image shape: {images.shape}, Mask shape: {masks.shape}")
                print(f"Image dtype: {images.dtype}, Mask dtype: {masks.dtype}")
                print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
                print(f"Mask range: [{masks.min():.3f}, {masks.max():.3f}]")
                
                # 保存原始图像和mask
                for j in range(images.shape[0]):
                    try:
                        # 保存图像
                        img_path = os.path.join(save_dir, f"batch{i}_sample{j}_image.png")
                        save_tensor_as_image(images[j], img_path)
                        
                        # 保存mask
                        mask_path = os.path.join(save_dir, f"batch{i}_sample{j}_mask.png")
                        save_tensor_as_image(masks[j], mask_path)
                        
                    except Exception as e:
                        print(f"Error saving sample {j} in batch {i}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error processing segmentation dataset {dataset_name}: {e}")
            continue

    # ========== 分类数据集可视化 ==========
    cls_datasets = {
        'Appendix': 2,
        'BUS-BRA': 2,
        'BUSI': 2,
        'Fatty-Liver': 2,
        'private_Appendix': 2,
        'private_Breast': 2,
        'private_Breast_luminal': 4,
        'private_Liver': 2
    }

    for dataset_name, num_classes in cls_datasets.items():
        dataset_path = os.path.join(args.root_path, "classification", dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Classification dataset not found: {dataset_path}")
            continue

        try:
            # 创建数据集和DataLoader
            db_cls = USdatasetOmni_cls_decoders(
                base_dir=dataset_path,
                split="train",
                transform=RandomGenerator_Cls(output_size=[args.img_size, args.img_size]),
                prompt=args.prompt
            )
            
            dataloader = DataLoader(db_cls, batch_size=4, shuffle=True, num_workers=4)
            
            # 创建保存目录
            save_dir = os.path.join(snapshot_path, "visualization", "classification", dataset_name)
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"\nVisualizing classification dataset: {dataset_name}")
            print(f"Dataset size: {len(db_cls)}, Num classes: {num_classes}")
            
            for i, batch in enumerate(tqdm(dataloader, desc=f"Processing {dataset_name}")):
                if i >= 3:  # 每个数据集只可视化3个batch
                    break
                    
                images = batch['image']
                labels = batch['label']
                
                print(f"Batch {i}: Image shape: {images.shape}, Label shape: {labels.shape}")
                print(f"Image dtype: {images.dtype}, Label dtype: {labels.dtype}")
                print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
                print(f"Labels: {labels.tolist()}")
                
                # 保存原始图像和标签
                for j in range(images.shape[0]):
                    try:
                        # 保存图像
                        img_path = os.path.join(save_dir, f"batch{i}_sample{j}_class{labels[j].item()}_image.png")
                        save_tensor_as_image(images[j], img_path)
                        
                        # 保存label信息
                        label_info_path = os.path.join(save_dir, f"batch{i}_sample{j}_class{labels[j].item()}_info.txt")
                        with open(label_info_path, 'w') as f:
                            f.write(f"Dataset: {dataset_name}\n")
                            f.write(f"Num classes: {num_classes}\n")
                            f.write(f"Label: {labels[j].item()}\n")
                            f.write(f"Image shape: {images[j].shape}\n")
                            if args.prompt and 'position_prompt' in batch:
                                f.write(f"Position prompt: {batch['position_prompt'][j]}\n")
                                f.write(f"Task prompt: {batch['task_prompt'][j]}\n")
                                f.write(f"Type prompt: {batch['type_prompt'][j]}\n")
                                f.write(f"Nature prompt: {batch['nature_prompt'][j]}\n")
                        
                    except Exception as e:
                        print(f"Error saving sample {j} in batch {i}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error processing classification dataset {dataset_name}: {e}")
            continue

    print("\nVisualization completed! Results saved to:", os.path.join(snapshot_path, "visualization"))

# 使用示例
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data', help='Root path for datasets')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--prompt', action='store_true', help='Whether to use prompt')
    args = parser.parse_args()
    
    visualize_input_images(args, "/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/vis_out")