#!/bin/bash

# ====== TU训练脚本（带Mixup和CutMix数据增强） ======

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用GPU 0和1
# export NCCL_DEBUG=INFO  # 启用NCCL调试信息

# 数据路径配置
DATA_ROOT="/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data"  # 请修改为您的数据路径
OUTPUT_DIR="./exp_out/TU_mixup"  # 输出目录

# 训练参数
MAX_EPOCHS=800
BATCH_SIZE=128
BASE_LR=0.00001
IMG_SIZE=224
SEED=1234

# Mixup和CutMix参数
MIXUP_ALPHA=0.8
CUTMIX_ALPHA=1.0
MIXUP_PROB=1.0
CUTMIX_PROB=1.0
MIXUP_SWITCH_PROB=0.5

# 模型配置
VIT_NAME="R50-ViT-B_16"
N_CLASSES_SEG=2

# 预训练模型路径（可选）
PRETRAIN_CKPT=""  # 如果有预训练模型，请设置路径

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "=== TU训练启动（带Mixup和CutMix） ==="
echo "数据路径: $DATA_ROOT"
echo "输出目录: $OUTPUT_DIR"
echo "最大训练轮数: $MAX_EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "基础学习率: $BASE_LR"
echo "图像尺寸: $IMG_SIZE"
echo "随机种子: $SEED"
echo ""

echo "=== Mixup/CutMix配置 ==="
echo "Mixup Alpha: $MIXUP_ALPHA"
echo "CutMix Alpha: $CUTMIX_ALPHA"
echo "Mixup概率: $MIXUP_PROB"
echo "CutMix概率: $CUTMIX_PROB"
echo "切换概率: $MIXUP_SWITCH_PROB"
echo ""

# 启动分布式训练
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    omni_train_TU_v2_mix.py \
    --root_path $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --base_lr $BASE_LR \
    --img_size $IMG_SIZE \
    --seed $SEED \
    --vit_name $VIT_NAME \
    --n_classes_seg $N_CLASSES_SEG \
    --mixup_alpha $MIXUP_ALPHA \
    --cutmix_alpha $CUTMIX_ALPHA \
    --mixup_prob $MIXUP_PROB \
    --cutmix_prob $CUTMIX_PROB \
    --mixup_switch_prob $MIXUP_SWITCH_PROB \
    --deterministic 1