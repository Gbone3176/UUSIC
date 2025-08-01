#!/usr/bin/env python3
"""
Script to analyze decoder parameters in OmniVisionTransformer PG_02
Author: Generated for parameter analysis
Date: 2025-07-30
"""

import sys
import os
import torch
import torch.nn as nn
import logging
sys.path.append('/storage/challenge-main/baseline')
from networks.omni_vision_transformer_PG_02 import OmniVisionTransformer
from config_PG import get_config
import argparse
from collections import defaultdict

def setup_logging():
    """
    设置logging配置
    """
    # 创建logs目录
    os.makedirs('logs', exist_ok=True)
    
    # 配置logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler('logs/network_parameters.txt', mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def format_params(params):
    """
    labels: Dict[str, int] = {"background": 0, "target": 1},
    """
    if params == 0:
        return "0"
    elif params < 1000:
        return f"{params}"
    elif params < 1_000_000:
        return f"{params/1000:.1f}K"
    elif params < 1_000_000_000:
        return f"{params/1_000_000:.1f}M"
    else:
        return f"{params/1_000_000_000:.1f}B"


def count_parameters(model, param_name_filter=None):
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        param_name_filter: 参数名过滤函数，返回True表示计算该参数
    
    Returns:
        total_params: 总参数数量
        param_dict: 各组件参数详情
    """
    total_params = 0
    param_dict = defaultdict(int)
    param_details = []
    
    for name, param in model.named_parameters():
        if param_name_filter is None or param_name_filter(name):
            param_count = param.numel()
            total_params += param_count
            
            # 分类统计 - 适配PG_02网络结构
            if 'seg_decoders' in name:
                param_dict['seg_decoders'] += param_count
            elif 'cls_decoders' in name:
                param_dict['cls_decoders'] += param_count
            elif 'seg_heads' in name:
                param_dict['seg_heads'] += param_count
            elif 'cls_heads' in name:
                param_dict['cls_heads'] += param_count
            elif 'seg_skip_connections' in name:
                param_dict['seg_skip_connections'] += param_count
            elif 'cls_skip_connections' in name:
                param_dict['cls_skip_connections'] += param_count
            elif 'norm_task_seg' in name:
                param_dict['seg_norm'] += param_count
            elif 'norm_task_cls' in name:
                param_dict['cls_norm'] += param_count
            else:
                param_dict['other'] += param_count
            
            param_details.append({
                'name': name,
                'shape': list(param.shape),
                'params': param_count
            })
    
    return total_params, param_dict, param_details


def get_dataset_specific_params(model, dataset_name, dataset_type):
    """
    获取特定数据集的参数，避免前缀匹配问题
    
    Args:
        model: PyTorch模型
        dataset_name: 数据集名称
        dataset_type: 数据集类型 ('seg' 或 'cls')
    
    Returns:
        dataset_params: 该数据集的参数数量
    """
    dataset_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        
        # 使用精确匹配，避免前缀问题
        if dataset_type == 'seg':
            if f'seg_decoders.{dataset_name}.' in name or f'seg_decoders.{dataset_name}' == name:
                dataset_params += param_count
            elif f'seg_heads.{dataset_name}.' in name or f'seg_heads.{dataset_name}' == name:
                dataset_params += param_count
            elif f'seg_skip_connections.{dataset_name}.' in name or f'seg_skip_connections.{dataset_name}' == name:
                dataset_params += param_count
        elif dataset_type == 'cls':
            if f'cls_decoders.{dataset_name}.' in name or f'cls_decoders.{dataset_name}' == name:
                dataset_params += param_count
            elif f'cls_heads.{dataset_name}.' in name or f'cls_heads.{dataset_name}' == name:
                dataset_params += param_count
            elif f'cls_skip_connections.{dataset_name}.' in name or f'cls_skip_connections.{dataset_name}' == name:
                dataset_params += param_count
    
    return dataset_params


def analyze_dataset_specific_decoders(model, logger):
    """
    分析数据集特定的decoder参数
    """
    logger.info("=" * 80)
    logger.info("DATASET-SPECIFIC DECODER ANALYSIS")
    logger.info("=" * 80)
    
    swin = model.swin
    
    # 分割数据集
    seg_datasets = swin.seg_datasets
    logger.info(f"\nSegmentation Datasets ({len(seg_datasets)}): {seg_datasets}")
    
    seg_total_params = 0
    for dataset_name in seg_datasets:
        # 使用精确匹配函数
        dataset_params = get_dataset_specific_params(model, dataset_name, 'seg')
        
        seg_total_params += dataset_params
        logger.info(f"  {dataset_name}: {format_params(dataset_params):>8} ({dataset_params:,}) params")
    
    logger.info(f"  Total Segmentation Decoders: {format_params(seg_total_params):>8} ({seg_total_params:,}) params")
    
    # 分类数据集
    cls_datasets = swin.cls_datasets
    logger.info(f"\nClassification Datasets ({len(cls_datasets)}): {cls_datasets}")
    
    cls_total_params = 0
    for dataset_name in cls_datasets:
        # 使用精确匹配函数
        dataset_params = get_dataset_specific_params(model, dataset_name, 'cls')
        
        cls_total_params += dataset_params
        logger.info(f"  {dataset_name}: {format_params(dataset_params):>8} ({dataset_params:,}) params")
    
    logger.info(f"  Total Classification Decoders: {format_params(cls_total_params):>8} ({cls_total_params:,}) params")
    
    return seg_total_params, cls_total_params


def analyze_decoder_components(model, logger):
    """
    详细分析decoder各组件的参数
    """
    logger.info("=" * 80)
    logger.info("DECODER COMPONENTS ANALYSIS")
    logger.info("=" * 80)
    
    # 分割decoder组件
    seg_components = {
        'seg_decoders': 'Segmentation Decoders',
        'seg_heads': 'Segmentation Output Heads',
        'seg_skip_connections': 'Segmentation Skip Connections',
        'norm_task_seg': 'Segmentation Normalization'
    }
    
    # 分类decoder组件
    cls_components = {
        'cls_decoders': 'Classification Decoders',
        'cls_heads': 'Classification Output Heads',
        'cls_skip_connections': 'Classification Skip Connections',
        'norm_task_cls': 'Classification Normalization'
    }
    
    all_components = {**seg_components, **cls_components}
    
    component_stats = {}
    
    for component_key, component_name in all_components.items():
        total_params, _, param_details = count_parameters(
            model, 
            lambda name: component_key in name
        )
        
        component_stats[component_key] = {
            'name': component_name,
            'total_params': total_params,
            'details': param_details
        }
        
        logger.info(f"\n{component_name}:")
        logger.info(f"  Total Parameters: {total_params}")
        
        if param_details:
            logger.info("  Layer Details:")
            for detail in param_details:
                logger.info(f"    {detail['name']}: {detail['shape']} -> {detail['params']} params")
        else:
            logger.info("    No parameters found")
    
    return component_stats


def analyze_decoder_layers_detail(model, logger):
    """
    分析decoder中每一层的详细参数
    """
    logger.info("\n" + "=" * 80)
    logger.info("DECODER LAYERS DETAILED ANALYSIS")
    logger.info("=" * 80)
    
    # 获取模型的swin组件
    swin = model.swin
    
    logger.info(f"Number of decoder layers: {swin.num_layers}")
    
    # 分析分割decoder各层
    logger.info("\n[SEGMENTATION DECODER LAYERS]")
    for dataset_name in swin.seg_datasets:
        logger.info(f"\nDataset: {dataset_name}")
        
        if dataset_name in swin.seg_decoders:
            seg_decoder = swin.seg_decoders[dataset_name]
            seg_skip = swin.seg_skip_connections[dataset_name]
            seg_head = swin.seg_heads[dataset_name]
            
            for i in range(swin.num_layers):
                logger.info(f"  Layer {i}:")
                
                # Skip connection layer
                if i < len(seg_skip):
                    skip_layer = seg_skip[i]
                    skip_params = sum(p.numel() for p in skip_layer.parameters())
                    logger.info(f"    Skip [{i}]:     {format_params(skip_params):>8} ({skip_params:,}) params - {type(skip_layer).__name__}")
                
                # Decoder layer
                if i < len(seg_decoder):
                    decoder_layer = seg_decoder[i]
                    decoder_params = sum(p.numel() for p in decoder_layer.parameters())
                    logger.info(f"    Decoder [{i}]:   {format_params(decoder_params):>8} ({decoder_params:,}) params - {type(decoder_layer).__name__}")
            
            # 分割输出头
            logger.info(f"  Head:")
            for i, head_layer in enumerate(seg_head):
                head_params = sum(p.numel() for p in head_layer.parameters())
                logger.info(f"    Head [{i}]:     {format_params(head_params):>8} ({head_params:,}) params - {type(head_layer).__name__}")
    
    # 分析分类decoder各层
    logger.info(f"\n[CLASSIFICATION DECODER LAYERS]")
    for dataset_name in swin.cls_datasets:
        logger.info(f"\nDataset: {dataset_name}")
        
        if dataset_name in swin.cls_decoders:
            cls_decoder = swin.cls_decoders[dataset_name]
            cls_skip = swin.cls_skip_connections[dataset_name]
            
            for i in range(swin.num_layers):
                logger.info(f"  Layer {i}:")
                
                # Skip connection layer  
                if i < len(cls_skip):
                    skip_layer = cls_skip[i]
                    skip_params = sum(p.numel() for p in skip_layer.parameters())
                    logger.info(f"    Skip [{i}]:     {format_params(skip_params):>8} ({skip_params:,}) params - {type(skip_layer).__name__}")
                
                # Decoder layer
                if i < len(cls_decoder):
                    decoder_layer = cls_decoder[i]
                    decoder_params = sum(p.numel() for p in decoder_layer.parameters())
                    logger.info(f"    Decoder [{i}]:   {format_params(decoder_params):>8} ({decoder_params:,}) params - {type(decoder_layer).__name__}")
            
            # 分类输出头
            logger.info(f"  Heads:")
            for head_key in [f"{dataset_name}_2cls", f"{dataset_name}_4cls"]:
                if head_key in swin.cls_heads:
                    head_layer = swin.cls_heads[head_key]
                    head_params = sum(p.numel() for p in head_layer.parameters())
                    logger.info(f"    {head_key}: {format_params(head_params):>8} ({head_params:,}) params - {type(head_layer).__name__}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Decoder Parameters')
    parser.add_argument('--root_path', type=str,
                        default='data/', help='root dir for data')
    parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument('--max_epochs', type=int,
                        default=300, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size per gpu')
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--cfg', type=str, default="configs/swin_tiny_patch4_window7_224_lite-PG.yaml",
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                        'full: cache all data, '
                        'part: sharding the dataset into non-overlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--pretrain_ckpt', type=str, help='pretrained checkpoint')

    parser.add_argument('--prompt', action='store_true', help='using prompt for training')
    parser.add_argument('--adapter_ft', action='store_true', help='using adapter for fine-tuning')
    parser.add_argument('--detailed', action='store_true', help='output detailed decoder component analysis')
    parser.add_argument('--layer_detail', action='store_true', help='output detailed layer analysis')
    args = parser.parse_args()
    
    # 设置logging
    logger = setup_logging()
    
    # 获取配置
    config = get_config(args)
    
    logger.info("=" * 80)
    logger.info("OMNI VISION TRANSFORMER PG_02 DECODER PARAMETER ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Config: {args.cfg}")
    logger.info(f"Prompt Mode: {args.prompt}")
    logger.info(f"Image Size: {config.DATA.IMG_SIZE}")
    logger.info(f"Embed Dim: {config.MODEL.SWIN.EMBED_DIM}")
    logger.info(f"Encoder Depths: {config.MODEL.SWIN.DEPTHS}")
    logger.info(f"Num Heads: {config.MODEL.SWIN.NUM_HEADS}")
    
    # 创建模型
    model = OmniVisionTransformer(config, prompt=args.prompt)
    
    # 1. 计算总参数
    total_params, _, _ = count_parameters(model)
    logger.info(f"\nTotal Model Parameters: {total_params}")
    
    # 2. 计算encoder参数 
    encoder_filter = lambda name: any(key in name for key in [
        'patch_embed', 'layers.', 'norm.', 'absolute_pos_embed'
    ]) and not any(key in name for key in ['seg_decoders', 'cls_decoders', 'seg_heads', 'cls_heads', 'seg_skip_connections', 'cls_skip_connections'])
    
    encoder_params, _, _ = count_parameters(model, encoder_filter)
    
    # 3. 计算decoder参数
    decoder_filter = lambda name: any(key in name for key in [
        'seg_decoders', 'cls_decoders', 'seg_heads', 'cls_heads', 
        'seg_skip_connections', 'cls_skip_connections', 'norm_task_seg', 'norm_task_cls'
    ])
    
    decoder_params, decoder_breakdown, decoder_details = count_parameters(model, decoder_filter)
    
    # 4. 分析数据集特定的decoder
    seg_total_params, cls_total_params = analyze_dataset_specific_decoders(model, logger)
    
    # 5. 输出统计结果
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER DISTRIBUTION SUMMARY")
    logger.info("=" * 80)
    
    decoder_ratio = (decoder_params / total_params) * 100
    encoder_ratio = (encoder_params / total_params) * 100
    other_ratio = 100 - decoder_ratio - encoder_ratio
    
    logger.info(f"Encoder Parameters:    {format_params(encoder_params):>8} ({encoder_params:>12,}) ({encoder_ratio:>6.2f}%)")
    logger.info(f"Decoder Parameters:    {format_params(decoder_params):>8} ({decoder_params:>12,}) ({decoder_ratio:>6.2f}%)")
    logger.info(f"Other Parameters:      {format_params(total_params - encoder_params - decoder_params):>8} ({total_params - encoder_params - decoder_params:>12,}) ({other_ratio:>6.2f}%)")
    logger.info(f"Total Parameters:      {format_params(total_params):>8} ({total_params:>12,}) ({100.0:>6.2f}%)")
    
    logger.info("\nDecoder Breakdown:")
    seg_decoder_params = decoder_breakdown['seg_decoders'] + decoder_breakdown['seg_heads'] + decoder_breakdown['seg_skip_connections'] + decoder_breakdown['seg_norm']
    cls_decoder_params = decoder_breakdown['cls_decoders'] + decoder_breakdown['cls_heads'] + decoder_breakdown['cls_skip_connections'] + decoder_breakdown['cls_norm']
    
    seg_ratio = (seg_decoder_params / total_params) * 100
    cls_ratio = (cls_decoder_params / total_params) * 100
    
    logger.info(f"  Segmentation Decoders: {format_params(seg_decoder_params):>8} ({seg_decoder_params:>12,}) ({seg_ratio:>6.2f}%)")
    logger.info(f"  Classification Decoders: {format_params(cls_decoder_params):>8} ({cls_decoder_params:>12,}) ({cls_ratio:>6.2f}%)")
    
    # 6. 详细组件分析
    if args.detailed:
        component_stats = analyze_decoder_components(model, logger)
        
        logger.info("\n" + "=" * 80)
        logger.info("DETAILED DECODER ANALYSIS")
        logger.info("=" * 80)
        
        # 按参数数量排序
        sorted_components = sorted(
            component_stats.items(), 
            key=lambda x: x[1]['total_params'], 
            reverse=True
        )
        
        logger.info("\nComponents ranked by parameter count:")
        for i, (key, stats) in enumerate(sorted_components, 1):
            ratio = (stats['total_params'] / total_params) * 100
            logger.info(f"{i:2d}. {stats['name']:<35} {format_params(stats['total_params']):>8} ({stats['total_params']:>10,}) ({ratio:>5.2f}%)")
    
    # 7. 层级详细分析
    if args.layer_detail:
        analyze_decoder_layers_detail(model, logger)
    
    # 8. 输出prompt相关参数（如果启用）
    if args.prompt:
        prompt_filter = lambda name: 'prompt' in name or 'hyper_param' in name
        prompt_params, _, prompt_details = count_parameters(model, prompt_filter)
        
        if prompt_params > 0:
            prompt_ratio = (prompt_params / total_params) * 100
            logger.info(f"Prompt Parameters:     {format_params(prompt_params):>8} ({prompt_params:>12,}) ({prompt_ratio:>6.2f}%)")
            if args.detailed and prompt_details:
                logger.info("\nPrompt Parameter Details:")
                for detail in prompt_details:
                    logger.info(f"  {detail['name']}: {detail['shape']} -> {detail['params']} params")
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
