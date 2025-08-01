# PG_decoder测试文件使用说明

## 概述

`omni_test_PG_decoders.py` 是专门为PG_decoder模型设计的测试文件，基于原有的 `omni_test_PG.py` 文件进行了适配和优化。

## 主要特点

### 1. 数据集特定的独立Decoder
- 支持为每个数据集使用独立的decoder进行推理
- 通过 `use_dataset_specific=True` 参数启用数据集特定功能
- 支持分割任务 (`task_type='seg'`) 和分类任务 (`task_type='cls'`)

### 2. 支持的数据集

#### 分割数据集
- `private_Thyroid`
- `private_Kidney` 
- `private_Fetal_Head`
- `private_Cardiac`
- `private_Breast_luminal`
- `private_Breast`

#### 分类数据集
- `private_Liver` (2分类)
- `private_Breast_luminal` (4分类)
- `private_Breast` (2分类)
- `private_Appendix` (2分类)

### 3. 模型导入
```python
from networks.omni_vision_transformer_PG_02 import OmniVisionTransformer as ViT_omni_decoders
```

## 使用方法

### 基本用法
```bash
python omni_test_PG_decoders.py \
    --root_path ./data \
    --output_dir ./test_results \
    --resume ./pretrained_ckpt/best_model.pth \
    --batch_size 1 \
    --img_size 224 \
    --cfg configs/swin_tiny_patch4_window7_224_lite-PG.yaml
```

### 使用Prompt的测试
```bash
python omni_test_PG_decoders.py \
    --root_path ./data \
    --output_dir ./test_results \
    --resume ./pretrained_ckpt/best_model.pth \
    --batch_size 1 \
    --img_size 224 \
    --cfg configs/swin_tiny_patch4_window7_224_lite-PG.yaml \
    --prompt
```

### 保存预测结果
```bash
python omni_test_PG_decoders.py \
    --root_path ./data \
    --output_dir ./test_results \
    --resume ./pretrained_ckpt/best_model.pth \
    --batch_size 1 \
    --img_size 224 \
    --cfg configs/swin_tiny_patch4_window7_224_lite-PG.yaml \
    --prompt \
    --is_saveout
```

## 关键参数说明

- `--root_path`: 数据根目录
- `--output_dir`: 输出目录，用于保存测试结果
- `--resume`: 预训练模型路径（必需）
- `--batch_size`: 批次大小，建议设为1
- `--img_size`: 输入图像尺寸
- `--cfg`: 配置文件路径
- `--prompt`: 是否使用prompt功能
- `--is_saveout`: 是否保存预测结果图像

## 输出结果

### 1. 测试日志
- `test_result.txt`: 详细的测试日志
- `result.csv`: 结构化的测试结果

### 2. 预测图像（当使用 `--is_saveout` 时）
- `{case_name}_pred.png`: 预测结果
- `{case_name}_img.png`: 原始图像
- `{case_name}_gt.png`: 真实标签

### 3. 性能指标
- **分割任务**: Dice系数
- **分类任务**: 准确率 (Accuracy)

## 与原始测试文件的区别

1. **模型导入**: 使用 `omni_vision_transformer_PG_02` 而不是 `omni_vision_transformer_PG`
2. **数据集特定推理**: 在模型调用时添加了 `use_dataset_specific=True` 参数
3. **测试数据集**: 专注于private数据集，移除了public数据集
4. **结果标识**: 在CSV结果中使用 `omni_seg_decoders` 和 `omni_cls_decoders` 标识

## 注意事项

1. 确保预训练模型是使用PG_decoder训练的
2. 数据路径结构必须正确
3. 配置文件必须与训练时使用的配置一致
4. 如果使用prompt，确保数据集中包含相应的prompt信息

## 故障排除

### 常见错误
1. **模型加载失败**: 检查模型路径和模型格式
2. **数据集路径错误**: 确认数据目录结构
3. **CUDA内存不足**: 减少batch_size或使用更小的图像尺寸
4. **配置文件不匹配**: 确保使用与训练时相同的配置文件

### 调试建议
1. 首先使用小数据集测试
2. 检查日志文件中的详细错误信息
3. 确认所有依赖包已正确安装
4. 验证GPU内存是否充足 