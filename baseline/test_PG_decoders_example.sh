#!/bin/bash

# PG_decoder测试示例脚本
# 使用方法: bash test_PG_decoders_example.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export LOCAL_RANK=0
export WORLD_SIZE=1

# 创建输出目录
OUTPUT_DIR="./test_results_PG_decoders"
mkdir -p $OUTPUT_DIR

# 运行PG_decoder测试
python omni_test_PG_decoders.py \
    --root_path ./data \
    --output_dir $OUTPUT_DIR \
    --resume ./pretrained_ckpt/best_model_PG_decoders.pth \
    --batch_size 1 \
    --img_size 224 \
    --cfg configs/swin_tiny_patch4_window7_224_lite-PG.yaml \
    --prompt \
    --is_saveout

echo "PG_decoder测试完成！结果保存在: $OUTPUT_DIR" 