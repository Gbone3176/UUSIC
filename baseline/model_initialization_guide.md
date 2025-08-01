# 模型初始化配置指南

## 修改后的模型需要的初始化配置

由于您对模型进行了修改，添加了数据集特定的decoder，确实需要在初始化时进行相应的配置。以下是需要关注的几个方面：

## 1. 数据集配置一致性

### 问题描述
模型中定义了支持的数据集列表，训练器中也定义了相应的数据集配置。这两者必须保持一致，否则会导致运行时错误。

### 当前配置状态

**模型中的配置** (`omni_vision_transformer_PG_02.py`):
```python
# 分割数据集
self.seg_datasets = ['private_Breast', 'private_Breast_luminal', 'private_Cardiac', 
                    'private_Fetal_Head', 'private_Kidney', 'private_Thyroid']

# 分类数据集  
self.cls_datasets = ['private_Appendix', 'private_Breast', 'private_Breast_luminal', 'private_Liver']
```

**训练器中的配置** (`omni_trainer_PG_decoders.py`):
```python
# 分割数据集
seg_datasets = [
    'private_Breast',
    'private_Breast_luminal', 
    'private_Cardiac',
    'private_Fetal_Head',
    'private_Kidney',
    'private_Thyroid'
]

# 分类数据集及其分类数
cls_datasets = {
    'private_Appendix': 2,
    'private_Breast': 2,
    'private_Breast_luminal': 4,  # 4分类
    'private_Liver': 4            # 4分类
}
```

## 2. 初始化验证机制

### 添加的验证功能
在 `omni_train_PG_decoders.py` 中添加了自动验证机制：

```python
def validate_dataset_config_local(model):
    # 验证分割数据集一致性
    # 验证分类数据集一致性  
    # 验证分类头是否正确创建
    return is_valid, validation_message
```

### 验证内容
1. **分割数据集列表**：模型和训练器中的分割数据集列表必须完全一致
2. **分类数据集列表**：模型和训练器中的分类数据集列表必须完全一致
3. **分类头存在性**：每个分类数据集的对应分类头必须在模型中正确创建

## 3. 模型结构验证

### 自动创建的组件
模型初始化时会自动为每个数据集创建：

**分割任务**：
- `seg_decoders[dataset_name]`: 数据集特定的分割decoder
- `seg_skip_connections[dataset_name]`: 对应的跳跃连接
- `seg_heads[dataset_name]`: 分割输出头

**分类任务**：
- `cls_decoders[dataset_name]`: 数据集特定的分类decoder
- `cls_skip_connections[dataset_name]`: 对应的跳跃连接
- `cls_heads[f"{dataset_name}_{num_classes}cls"]`: 分类输出头

### 输出验证信息
```bash
Model segmentation datasets: ['private_Breast', 'private_Breast_luminal', ...]
Model classification datasets: ['private_Appendix', 'private_Breast', ...]
Model segmentation decoders: ['private_Breast', 'private_Breast_luminal', ...]
Model classification decoders: ['private_Appendix', 'private_Breast', ...]
Model classification heads: ['private_Appendix_2cls', 'private_Breast_2cls', ...]
```

## 4. 预训练权重加载

### 注意事项
由于添加了新的数据集特定decoder，预训练权重加载时：

1. **encoder部分**：可以正常加载原有的预训练权重
2. **decoder部分**：新添加的数据集特定decoder会随机初始化
3. **分类头**：新的分类头也会随机初始化

### 建议的加载策略
```python
if args.pretrain_ckpt is not None:
    # 加载预训练权重，新添加的decoder会自动跳过
    net.load_from_self(args.pretrain_ckpt)
    print("Loaded pretrained weights. New dataset-specific decoders are randomly initialized.")
else:
    # 从config加载（通常是ImageNet预训练的encoder）
    net.load_from(config)
    print("Loaded encoder weights from config. All decoders are randomly initialized.")
```

## 5. 数据目录检查

### 新增功能
添加了 `--skip_missing_datasets` 参数：
- 当设置时，会跳过不存在的数据集目录
- 未设置时，缺少数据集会报错

### 使用方法
```bash
# 跳过缺失的数据集
python omni_train_PG_decoders.py --skip_missing_datasets

# 严格检查所有数据集必须存在
python omni_train_PG_decoders.py
```

## 6. 常见初始化问题及解决方案

### 问题1：数据集配置不匹配
**错误信息**：`Dataset configuration validation failed: Classification datasets mismatch`

**解决方案**：
1. 检查模型中的 `seg_datasets` 和 `cls_datasets` 定义
2. 检查训练器中对应的数据集配置
3. 确保两者完全一致

### 问题2：分类头缺失
**错误信息**：`Missing classification head: private_Liver_4cls`

**解决方案**：
1. 检查模型初始化时是否正确创建了所有分类头
2. 确认训练器中的分类数配置正确

### 问题3：数据目录不存在
**错误信息**：`Failed to create DataLoader for dataset_name`

**解决方案**：
1. 确保数据目录结构正确
2. 使用 `--skip_missing_datasets` 跳过缺失数据集
3. 或者创建对应的数据目录

## 7. 初始化检查清单

在运行训练前，请确认：

- [ ] 模型和训练器中的数据集配置一致
- [ ] 所有需要的数据目录存在
- [ ] 预训练权重路径正确（如果使用）
- [ ] 配置文件路径正确
- [ ] 输出目录有写入权限

## 8. 调试技巧

### 启用验证信息
运行时会自动显示详细的配置信息，包括：
- 支持的数据集列表
- 创建的decoder列表
- 创建的分类头列表

### 检查模型结构
```python
# 查看模型中创建的所有组件
print("Segmentation decoders:", list(net.swin.seg_decoders.keys()))
print("Classification decoders:", list(net.swin.cls_decoders.keys()))
print("Classification heads:", list(net.swin.cls_heads.keys()))
```

这样的初始化配置确保了模型能够正确处理所有指定的数据集，并提供了充分的验证和调试信息。
