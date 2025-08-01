"""
数据集分离训练方案说明

这个方案通过在数据加载阶段就按数据集分离，确保每个batch只包含来自同一数据集的数据，
从而避免了在forward函数中处理混合批次的复杂性。

主要修改点：
1. 为每个数据集创建独立的DataLoader
2. 修改训练循环，按数据集逐个进行训练
3. 通过权重控制各数据集的训练频率

修改详情：
"""

# ========== 原始方案 ==========
# 原来：使用混合数据集
# db_train_seg = USdatasetOmni_seg(...)  # 包含所有分割数据集
# trainloader_seg = DataLoader(db_train_seg, ...)

# 训练时：
# for batch in trainloader_seg:
#     dataset_name = batch['dataset_name']  # batch中可能包含不同数据集
#     output = model(..., dataset_name=dataset_name)  # 需要处理混合批次


# ========== 新方案 ==========
"""
1. 数据加载修改：
   - 为每个数据集创建独立的DataLoader
   - seg_dataloaders = {
       'private_Breast': {'loader': DataLoader(...), 'weight': 0.25, 'sampler': ...},
       'private_Cardiac': {'loader': DataLoader(...), 'weight': 4, 'sampler': ...},
       ...
     }
   - cls_dataloaders = {
       'private_Appendix': {'loader': DataLoader(...), 'weight': 4, 'num_classes': 2},
       'private_Liver': {'loader': DataLoader(...), 'weight': 2, 'num_classes': 4},
       ...
     }

2. 训练循环修改：
   # 分割任务训练
   for dataset_name, dataset_info in seg_dataloaders.items():
       dataloader = dataset_info['loader']
       weight = dataset_info['weight']
       
       for batch in dataloader:
           # 此时batch中所有数据都来自同一个数据集(dataset_name)
           output = model(..., dataset_name=dataset_name, task_type='seg')
           # 无需处理混合批次问题！
   
   # 分类任务训练  
   for dataset_name, dataset_info in cls_dataloaders.items():
       dataloader = dataset_info['loader']
       weight = dataset_info['weight']
       num_classes = dataset_info['num_classes']
       
       for batch in dataloader:
           # 同样，batch中所有数据都来自同一个数据集
           output = model(..., dataset_name=dataset_name, task_type='cls', num_classes=num_classes)

3. 权重控制：
   - 通过 skip_prob = max(0.0, 1.0 - weight) 控制各数据集的训练频率
   - 权重高的数据集训练更频繁，权重低的数据集会跳过一些批次

4. 优势：
   ✅ 无需修改model的forward函数
   ✅ 每个batch内数据同质，可以正常并行计算
   ✅ 通过权重控制各数据集的训练比例
   ✅ 代码结构清晰，易于调试和维护
   ✅ 可以为每个数据集设置不同的参数（如学习率、batch_size等）

5. 主要配置：
   
   # 分割数据集权重配置
   seg_dataset_weights = {
       'private_Breast': 0.25,        # 数据多，权重低
       'private_Breast_luminal': 0.25,
       'private_Cardiac': 4,          # 数据少，权重高  
       'private_Fetal_Head': 4,
       'private_Kidney': 2,
       'private_Thyroid': 4
   }
   
   # 分类数据集配置
   cls_datasets = {
       'private_Appendix': 2,          # 2分类
       'private_Breast': 2,            # 2分类
       'private_Breast_luminal': 4,    # 4分类
       'private_Liver': 4              # 4分类
   }
   
   cls_dataset_weights = {
       'private_Appendix': 4,
       'private_Breast': 4,
       'private_Breast_luminal': 1,
       'private_Liver': 2
   }

使用方法：
1. 确保数据按数据集名称组织在对应文件夹中
2. 调整权重配置以平衡各数据集的训练频率
3. 运行训练，每个epoch会按数据集顺序训练
4. 监控各数据集的训练损失

注意事项：
- 确保数据路径正确，数据集文件夹存在
- 根据实际数据分布调整权重
- 监控各数据集的收敛情况
- 可以根据需要调整每个数据集的worker数量和batch_size
"""

def print_training_summary():
    print("""
    ========== 训练流程总结 ==========
    
    每个Epoch的训练流程：
    1. 设置所有sampler的epoch
    2. 分割任务训练：
       - 遍历每个分割数据集
       - 对每个数据集使用其专用的DataLoader和decoder
       - 根据权重控制训练频率
    3. 分类任务训练：
       - 遍历每个分类数据集  
       - 对每个数据集使用其专用的DataLoader和decoder
       - 支持2分类和4分类
    4. 验证和模型保存
    
    关键优势：
    ✅ 解决了混合批次问题
    ✅ 保持了数据集特定decoder的优势
    ✅ 代码结构清晰，便于维护
    ✅ 支持灵活的权重配置
    """)

if __name__ == "__main__":
    print_training_summary()
