#!/usr/bin/env python3
"""
Script to analyze decoder parameters in OmniVisionTransformer
Author: Generated for parameter analysis
Date: 2025-07-30
"""

import sys
import os
import torch
import torch.nn as nn
sys.path.append('/storage/challenge-main/baseline')
from networks.omni_vision_transformer_PG import OmniVisionTransformer
from config_PG import get_config
import argparse
from collections import defaultdict
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
            
            # 分类统计
            if 'layers_task_seg' in name:
                param_dict['seg_decoder'] += param_count
            elif 'layers_task_cls' in name:
                param_dict['cls_decoder'] += param_count
            elif 'norm_task_seg' in name:
                param_dict['seg_decoder'] += param_count
            elif 'norm_task_cls' in name:
                param_dict['cls_decoder'] += param_count
            else:
                param_dict['other'] += param_count
            
            param_details.append({
                'name': name,
                'shape': list(param.shape),
                'params': param_count
            })
    
    return total_params, param_dict, param_details


def analyze_decoder_components(model):
    """
    详细分析decoder各组件的参数
    """
    print("=" * 80)
    print("DECODER COMPONENTS ANALYSIS")
    print("=" * 80)
    
    # 分割decoder组件
    seg_components = {
        'layers_task_seg_up': 'Segmentation Upsampling Layers',
        'layers_task_seg_skip': 'Segmentation Skip Connection Layers', 
        'layers_task_seg_head': 'Segmentation Output Head',
        'norm_task_seg': 'Segmentation Normalization'
    }
    
    # 分类decoder组件
    cls_components = {
        'layers_task_cls_up': 'Classification Upsampling Layers',
        'layers_task_cls_skip': 'Classification Skip Connection Layers',
        'layers_task_cls_head': 'Classification Output Head', 
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
        
        print(f"\n{component_name}:")
        print(f"  Total Parameters: {total_params}")
        
        if param_details:
            print("  Layer Details:")
            for detail in param_details:
                print(f"    {detail['name']}: {detail['shape']} -> {detail['params']} params")
        else:
            print("    No parameters found")
    
    return component_stats


def analyze_decoder_layers_detail(model):
    """
    分析decoder中每一层的详细参数
    """
    print("\n" + "=" * 80)
    print("DECODER LAYERS DETAILED ANALYSIS")
    print("=" * 80)
    
    # 获取模型的swin组件
    swin = model.swin
    
    print(f"Number of decoder layers: {swin.num_layers}")
    
    # 分析分割decoder各层
    print("\n[SEGMENTATION DECODER LAYERS]")
    for i in range(swin.num_layers):
        print(f"\nLayer {i}:")
        
        # Skip connection layer
        if i < len(swin.layers_task_seg_skip):
            skip_layer = swin.layers_task_seg_skip[i]
            skip_params = sum(p.numel() for p in skip_layer.parameters())
            print(f"  Seg Skip [{i}]:     {format_params(skip_params):>8} ({skip_params:,}) params - {type(skip_layer).__name__}")
        
        # Upsampling layer
        if i < len(swin.layers_task_seg_up):
            up_layer = swin.layers_task_seg_up[i]
            up_params = sum(p.numel() for p in up_layer.parameters())
            print(f"  Seg Up [{i}]:       {format_params(up_params):>8} ({up_params:,}) params - {type(up_layer).__name__}")
    
    # 分割输出头
    print(f"\n[SEGMENTATION HEAD]")
    for i, head_layer in enumerate(swin.layers_task_seg_head):
        head_params = sum(p.numel() for p in head_layer.parameters())
        print(f"  Seg Head [{i}]:     {format_params(head_params):>8} ({head_params:,}) params - {type(head_layer).__name__}")
    
    # 分析分类decoder各层
    print(f"\n[CLASSIFICATION DECODER LAYERS]")
    for i in range(swin.num_layers):
        print(f"\nLayer {i}:")
        
        # Skip connection layer  
        if i < len(swin.layers_task_cls_skip):
            skip_layer = swin.layers_task_cls_skip[i]
            skip_params = sum(p.numel() for p in skip_layer.parameters())
            print(f"  Cls Skip [{i}]:     {format_params(skip_params):>8} ({skip_params:,}) params - {type(skip_layer).__name__}")
        
        # Upsampling layer
        if i < len(swin.layers_task_cls_up):
            up_layer = swin.layers_task_cls_up[i]
            up_params = sum(p.numel() for p in up_layer.parameters())
            print(f"  Cls Up [{i}]:       {format_params(up_params):>8} ({up_params:,}) params - {type(up_layer).__name__}")
    
    # 分类输出头
    print(f"\n[CLASSIFICATION HEAD]")
    for i, head_layer in enumerate(swin.layers_task_cls_head):
        head_params = sum(p.numel() for p in head_layer.parameters())
        print(f"  Cls Head [{i}]:     {format_params(head_params):>8} ({head_params:,}) params - {type(head_layer).__name__}")


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
    
    # 获取配置
    config = get_config(args)
    
    print("=" * 80)
    print("OMNI VISION TRANSFORMER DECODER PARAMETER ANALYSIS")
    print("=" * 80)
    print(f"Config: {args.cfg}")
    print(f"Prompt Mode: {args.prompt}")
    print(f"Image Size: {config.DATA.IMG_SIZE}")
    print(f"Embed Dim: {config.MODEL.SWIN.EMBED_DIM}")
    print(f"Encoder Depths: {config.MODEL.SWIN.DEPTHS}")
    print(f"Num Heads: {config.MODEL.SWIN.NUM_HEADS}")
    
    # 创建模型
    model = OmniVisionTransformer(config, prompt=args.prompt)
    
    # 1. 计算总参数
    total_params, _, _ = count_parameters(model)
    print(f"\nTotal Model Parameters: {total_params}")
    
    # 2. 计算decoder参数
    decoder_filter = lambda name: any(key in name for key in [
        'layers_task_seg', 'layers_task_cls', 'norm_task_seg', 'norm_task_cls'
    ])
    
    decoder_params, decoder_breakdown, decoder_details = count_parameters(model, decoder_filter)
    
    # 3. 计算encoder参数 
    encoder_filter = lambda name: any(key in name for key in [
        'patch_embed', 'layers.', 'norm.', 'absolute_pos_embed'
    ]) and not any(key in name for key in ['layers_task'])
    
    encoder_params, _, _ = count_parameters(model, encoder_filter)
    
    # 4. 输出统计结果
    print("\n" + "=" * 80)
    print("PARAMETER DISTRIBUTION SUMMARY")
    print("=" * 80)
    
    decoder_ratio = (decoder_params / total_params) * 100
    encoder_ratio = (encoder_params / total_params) * 100
    other_ratio = 100 - decoder_ratio - encoder_ratio
    
    print(f"Encoder Parameters:    {format_params(encoder_params):>8} ({encoder_params:>12,}) ({encoder_ratio:>6.2f}%)")
    print(f"Decoder Parameters:    {format_params(decoder_params):>8} ({decoder_params:>12,}) ({decoder_ratio:>6.2f}%)")
    print(f"Other Parameters:      {format_params(total_params - encoder_params - decoder_params):>8} ({total_params - encoder_params - decoder_params:>12,}) ({other_ratio:>6.2f}%)")
    print(f"Total Parameters:      {format_params(total_params):>8} ({total_params:>12,}) ({100.0:>6.2f}%)")
    
    print("\nDecoder Breakdown:")
    seg_decoder_params = decoder_breakdown['seg_decoder']
    cls_decoder_params = decoder_breakdown['cls_decoder']
    
    seg_ratio = (seg_decoder_params / total_params) * 100
    cls_ratio = (cls_decoder_params / total_params) * 100
    
    print(f"  Segmentation Decoder: {format_params(seg_decoder_params):>8} ({seg_decoder_params:>12,}) ({seg_ratio:>6.2f}%)")
    print(f"  Classification Decoder: {format_params(cls_decoder_params):>8} ({cls_decoder_params:>12,}) ({cls_ratio:>6.2f}%)")
    
    # 5. 详细组件分析
    if args.detailed:
        component_stats = analyze_decoder_components(model)
        
        print("\n" + "=" * 80)
        print("DETAILED DECODER ANALYSIS")
        print("=" * 80)
        
        # 按参数数量排序
        sorted_components = sorted(
            component_stats.items(), 
            key=lambda x: x[1]['total_params'], 
            reverse=True
        )
        
        print("\nComponents ranked by parameter count:")
        for i, (key, stats) in enumerate(sorted_components, 1):
            ratio = (stats['total_params'] / total_params) * 100
            print(f"{i:2d}. {stats['name']:<35} {format_params(stats['total_params']):>8} ({stats['total_params']:>10,}) ({ratio:>5.2f}%)")
    
    # 6. 层级详细分析
    if args.layer_detail:
        analyze_decoder_layers_detail(model)
    
    # 7. 输出prompt相关参数（如果启用）
    if args.prompt:
        prompt_filter = lambda name: 'prompt' in name or 'hyper_param' in name
        prompt_params, _, prompt_details = count_parameters(model, prompt_filter)
        
        if prompt_params > 0:
            prompt_ratio = (prompt_params / total_params) * 100
            print(f"Prompt Parameters:     {format_params(prompt_params):>8} ({prompt_params:>12,}) ({prompt_ratio:>6.2f}%)")
            if args.detailed and prompt_details:
                print("\nPrompt Parameter Details:")
                for detail in prompt_details:
                    print(f"  {detail['name']}: {detail['shape']} -> {detail['params']} params")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
