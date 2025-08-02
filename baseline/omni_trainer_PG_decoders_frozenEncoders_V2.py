import os
import sys
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from utils import DiceLoss
from datasets.dataset_aug_norm import USdatasetCls, USdatasetSeg
from datasets.dataset_aug_norm import RandomGenerator_Cls, RandomGenerator_Seg, CenterCropGenerator
from datasets.omni_dataset_decoders import WeightedRandomSamplerDDP
from datasets.omni_dataset_decoders import USdatasetOmni_cls_decoders, USdatasetOmni_seg_decoders
from sklearn.metrics import roc_auc_score

from utils import omni_seg_test_decoders

# Warmup + CosineAnnealingLR调度器
import math
class WarmupCosineLRScheduler:
    def __init__(self, optimizer, base_lr, max_iters, warmup_iters=0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.last_iter = 0
        self.set_lr(0.0 if warmup_iters > 0 else base_lr)

    def step(self):
        self.last_iter += 1
        if self.last_iter <= self.warmup_iters and self.warmup_iters > 0:
            lr = self.base_lr * self.last_iter / float(self.warmup_iters)
        else:
            t = min(self.last_iter - self.warmup_iters, self.max_iters - self.warmup_iters)
            T = max(1, self.max_iters - self.warmup_iters)
            lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * t / T))
        self.set_lr(lr)
        return lr

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_lr = lr

    def get_lr(self):
        return self.current_lr

def validate_dataset_config(model):
    """
    验证模型和训练器中的数据集配置是否一致
    """
    # 训练器中的数据集配置
    trainer_seg_datasets = [
        'private_Breast',
        'private_Breast_luminal', 
        'private_Cardiac',
        'private_Fetal_Head',
        'private_Kidney',
        'private_Thyroid'
    ]
    
    trainer_cls_datasets = {
        'private_Appendix': 2,
        'private_Breast': 2,
        'private_Breast_luminal': 4,
        'private_Liver': 2
    }
    
    # 从模型中获取数据集配置
    model_seg_datasets = model.swin.seg_datasets
    model_cls_datasets = model.swin.cls_datasets
    
    # 验证分割数据集
    if set(trainer_seg_datasets) != set(model_seg_datasets):
        return False, f"Segmentation datasets mismatch: trainer={trainer_seg_datasets}, model={model_seg_datasets}"
    
    # 验证分类数据集
    if set(trainer_cls_datasets.keys()) != set(model_cls_datasets):
        return False, f"Classification datasets mismatch: trainer={list(trainer_cls_datasets.keys())}, model={model_cls_datasets}"
    
    # 验证分类数据集的分类头是否存在
    for dataset_name, num_classes in trainer_cls_datasets.items():
        head_key = f"{dataset_name}_{num_classes}cls"
        if head_key not in model.swin.cls_heads:
            return False, f"Missing classification head: {head_key}"
    
    return True, "All dataset configurations are consistent"


def omni_train(args, model, snapshot_path):

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gpu_id = rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(seconds=7200))

    if int(os.environ["LOCAL_RANK"]) == 0:
        print('** GPU NUM ** : ', torch.cuda.device_count())
        print('** WORLD SIZE ** : ', torch.distributed.get_world_size())
    print(f"** DDP ** : Start running on rank {rank}.")

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # ========== 修改为按数据集分别创建DataLoader ==========
    
    # ######### for debug #########
    # seg_datasets = [
    #     'private_Breast',
    # ]
    
    # # 定义分类任务的数据集列表和其分类数
    # cls_datasets = {
    #     'private_Appendix': 2,
    # }
    
    # # 为每个分割数据集创建单独的DataLoader
    # seg_dataloaders = {}
    # seg_dataset_weights = {
    #     'private_Breast': 0.25,
    # }
    
    # ######### for train #########
    seg_datasets = [
        'private_Breast',
        'private_Breast_luminal', 
        'private_Cardiac',
        'private_Fetal_Head',
        'private_Kidney',
        'private_Thyroid'
    ]
    
    # 定义分类任务的数据集列表和其分类数
    cls_datasets = {
        'private_Appendix': 2,
        'private_Breast': 2,
        'private_Breast_luminal': 4,  # 4分类
        'private_Liver': 2           
    }
    
    # 为每个分割数据集创建单独的DataLoader
    seg_dataloaders = {}
    # 改进的权重配置 - 基于数据平衡的考虑
    seg_dataset_weights = {
        'private_Breast': 1.0,          # 基准权重
        'private_Breast_luminal': 1.2,  # 略微增加
        'private_Cardiac': 1.8,         # 较大增加（通常数据较少）
        'private_Fetal_Head': 1.8,      # 较大增加（通常数据较少）
        'private_Kidney': 1.5,          # 中等增加
        'private_Thyroid': 2.0          # 最大增加（通常数据最少）
    }
    
    logging.info("Creating segmentation DataLoaders for each dataset...")
    for dataset_name in seg_datasets:
        dataset_path = os.path.join(args.root_path, "segmentation", dataset_name)
        if os.path.exists(dataset_path):
            try:

                db_seg = USdatasetOmni_seg_decoders(
                    base_dir=dataset_path,
                    split="train",
                    transform=transforms.Compose([RandomGenerator_Seg(output_size=[args.img_size, args.img_size])]),
                    prompt=args.prompt
                )
                
                # 使用DDP sampler
                sampler = torch.utils.data.distributed.DistributedSampler(
                    db_seg, 
                    num_replicas=world_size, 
                    rank=rank,
                    shuffle=True
                )
                
                dataloader = DataLoader(
                    db_seg,
                    batch_size=batch_size,
                    num_workers=12,  # 减少每个loader的worker数
                    pin_memory=True,
                    worker_init_fn=worker_init_fn,
                    sampler=sampler
                )
                
                seg_dataloaders[dataset_name] = {
                    'loader': dataloader,
                    'sampler': sampler,
                    'weight': seg_dataset_weights.get(dataset_name, 1.0),
                    'size': len(db_seg)
                }
                logging.info(f"Created segmentation DataLoader for {dataset_name}: {len(db_seg)} samples")
            except Exception as e:
                logging.warning(f"Failed to create DataLoader for {dataset_name}: {e}")
    
    # 为每个分类数据集创建单独的DataLoader
    cls_dataloaders = {}
    # 改进的权重配置 - 基于数据平衡和任务难度的考虑
    cls_dataset_weights = {
        'private_Appendix': 1.8,        # 增加权重（通常数据较少）
        'private_Breast': 1.0,          # 基准权重
        'private_Breast_luminal': 2.5,  # 大幅增加（4分类任务且数据可能较少）
        'private_Liver': 1.6            # 适度增加
    }
    
    logging.info("Creating classification DataLoaders for each dataset...")  
    for dataset_name, num_classes in cls_datasets.items():
        dataset_path = os.path.join(args.root_path, "classification", dataset_name)
        if os.path.exists(dataset_path):
            try:
                db_cls = USdatasetOmni_cls_decoders(
                    base_dir=dataset_path,
                    split="train",
                    transform=transforms.Compose([RandomGenerator_Cls(output_size=[args.img_size, args.img_size])]),
                    prompt=args.prompt
                )
                
                # 使用DDP sampler
                sampler = torch.utils.data.distributed.DistributedSampler(
                    db_cls,
                    num_replicas=world_size,
                    rank=rank, 
                    shuffle=True
                )
                
                dataloader = DataLoader(
                    db_cls,
                    batch_size=batch_size,
                    num_workers=8,  # 减少每个loader的worker数
                    pin_memory=True,
                    worker_init_fn=worker_init_fn,
                    sampler=sampler
                )
                
                cls_dataloaders[dataset_name] = {
                    'loader': dataloader,
                    'sampler': sampler,
                    'weight': cls_dataset_weights.get(dataset_name, 1.0),
                    'num_classes': num_classes,
                    'size': len(db_cls)
                }
                logging.info(f"Created classification DataLoader for {dataset_name}: {len(db_cls)} samples, {num_classes} classes")
            except Exception as e:
                logging.warning(f"Failed to create DataLoader for {dataset_name}: {e}")

    model = model.to(device=device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)

    model.train()

    seg_ce_loss = CrossEntropyLoss()
    seg_dice_loss = DiceLoss()
    # cls_ce_loss = CrossEntropyLoss()

    cls_ce_loss_2way = CrossEntropyLoss()
    cls_ce_loss_4way = CrossEntropyLoss()

    # optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01, betas=(0.9, 0.999))

    # resume_epoch = 0
    # if args.resume is not None:
    #     model.load_state_dict(torch.load(args.resume, map_location='cpu')['model'])
    #     optimizer.load_state_dict(torch.load(args.resume, map_location='cpu')['optimizer'])
    #     resume_epoch = torch.load(args.resume, map_location='cpu')['epoch']

    # ========== 修改的模型加载和权重冻结部分 ==========
    # 仅加载encoder权重并冻结
    resume_epoch = load_encoder_weights_and_freeze(model, args.resume)

    # 创建优化器（只优化可训练的参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=base_lr, weight_decay=0.01, betas=(0.9, 0.999))

    logging.info(f"Optimizer created with {len(trainable_params)} trainable parameter groups")


    writer = SummaryWriter(snapshot_path + '/log')
    global_iter_num = 0
    seg_iter_num = 0
    cls_iter_num = 0
    max_epoch = args.max_epochs

    # 计算总迭代次数
    total_seg_iterations = sum(len(info['loader']) for info in seg_dataloaders.values())
    total_cls_iterations = sum(len(info['loader']) for info in cls_dataloaders.values())
    total_iterations = total_seg_iterations + total_cls_iterations
    max_iterations = args.max_epochs * total_iterations

    # warmup_batch可通过args.warmup_batch指定，默认5
    warmup_batch = getattr(args, 'warmup_batch', 5)
    lr_scheduler = WarmupCosineLRScheduler(optimizer, base_lr, max_iterations, warmup_iters=warmup_batch)

    logging.info("{} batch size. {} total seg iterations + {} total cls iterations = {} total iterations per epoch. {} max iterations ".format(
        batch_size, total_seg_iterations, total_cls_iterations, total_iterations, max_iterations))
    best_performance = 0.0
    best_epoch = 0

    best_models = []  # 存储 (epoch, performance, model_path) 的列表
    max_saved_models = 10  # 保存最好的10个模型
    if int(os.environ["LOCAL_RANK"]) != 0:
        iterator = tqdm(range(resume_epoch, max_epoch), ncols=70, disable=True)
    else:
        iterator = tqdm(range(resume_epoch, max_epoch), ncols=70, disable=False)

    ## 添加早停机制
    patience = 10
    no_improve_count = 0
    best_performance = 0.0

    for epoch_num in iterator:
        logging.info("\n epoch: {}".format(epoch_num))
        
        # 为每个数据集的sampler设置epoch
        for dataset_name, info in seg_dataloaders.items():
            info['sampler'].set_epoch(epoch_num)
        for dataset_name, info in cls_dataloaders.items():
            info['sampler'].set_epoch(epoch_num)

        # ========== 分割任务训练 - 加权采样 ==========
        logging.info("Training segmentation tasks...")
        
        # 创建加权采样池
        seg_batch_pool = []
        for dataset_name, dataset_info in seg_dataloaders.items():
            dataloader = dataset_info['loader']
            weight = dataset_info['weight']
            
            # 根据权重决定每个数据集的采样次数
            sample_count = int(len(dataloader) * weight)
            sample_count = max(1, sample_count)  # 确保至少采样一次
            
            # logging.info(f"Dataset {dataset_name}: original batches={len(dataloader)}, weight={weight}, target samples={sample_count}")
            
            # 将批次添加到采样池，重复次数根据权重决定
            batch_list = list(enumerate(dataloader))
            if weight <= 1.0:
                # 权重<=1时，随机采样子集
                sampled_batches = random.sample(batch_list, min(sample_count, len(batch_list)))
            else:
                # 权重>1时，重复采样
                sampled_batches = random.choices(batch_list, k=sample_count)
            
            for i_batch, sampled_batch in sampled_batches:
                seg_batch_pool.append((dataset_name, i_batch, sampled_batch))
        
        # 随机打乱采样池
        random.shuffle(seg_batch_pool)
        logging.info(f"Total segmentation batches after weighted sampling: {len(seg_batch_pool)}")
        
        # 训练加权采样后的批次
        for dataset_name, i_batch, sampled_batch in tqdm(seg_batch_pool, total=len(seg_batch_pool), desc="Seg-Training"):
                   
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.to(device=device), label_batch.to(device=device)
                
                if args.prompt:
                    position_prompt = torch.stack(sampled_batch['position_prompt']).permute([1, 0]).float().to(device=device)
                    task_prompt = torch.stack(sampled_batch['task_prompt']).permute([1, 0]).float().to(device=device)
                    type_prompt = torch.stack(sampled_batch['type_prompt']).permute([1, 0]).float().to(device=device)
                    nature_prompt = torch.stack(sampled_batch['nature_prompt']).permute([1, 0]).float().to(device=device) 
                    
                    x_seg = model((image_batch, position_prompt, task_prompt, type_prompt, nature_prompt),
                                          use_dataset_specific=True, dataset_name=dataset_name, task_type='seg', num_classes=2)
                else:
                    x_seg = model(image_batch, use_dataset_specific=True, dataset_name=dataset_name, task_type='seg', num_classes=2)

                loss_ce = seg_ce_loss(x_seg, label_batch[:].long())
                loss_dice = seg_dice_loss(x_seg, label_batch, softmax=True)

                ce_weight, dice_weight = get_dynamic_loss_weights(epoch_num, max_epoch)
                loss = ce_weight * loss_ce + dice_weight * loss_dice

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                

                # 分割和分类公用同一个学习率调度器
                lr_ = lr_scheduler.step()

                seg_iter_num = seg_iter_num + 1
                global_iter_num = global_iter_num + 1

                writer.add_scalar('info/lr_seg', lr_, seg_iter_num)
                writer.add_scalar(f'info/seg_loss_{dataset_name}', loss, seg_iter_num)
                writer.add_scalar('info/seg_loss_total', loss, seg_iter_num)

                if global_iter_num % 50 == 0:  # 减少日志频率
                    logging.info('Dataset: %s, global iteration %d and seg iteration %d : loss : %f' %
                                 (dataset_name, global_iter_num, seg_iter_num, loss.item()))

        # ========== 分类任务训练 - 加权采样 ==========
        logging.info("Training classification tasks...")
        
        # 创建加权采样池
        cls_batch_pool = []
        for dataset_name, dataset_info in cls_dataloaders.items():
            dataloader = dataset_info['loader']
            weight = dataset_info['weight']
            num_classes = dataset_info['num_classes']
            
            # 根据权重决定每个数据集的采样次数
            sample_count = int(len(dataloader) * weight)
            sample_count = max(1, sample_count)  # 确保至少采样一次
            
            # logging.info(f"Dataset {dataset_name}: original batches={len(dataloader)}, weight={weight}, target samples={sample_count}")
            
            # 将批次添加到采样池，重复次数根据权重决定
            batch_list = list(enumerate(dataloader))
            if weight <= 1.0:
                # 权重<=1时，随机采样子集
                sampled_batches = random.sample(batch_list, min(sample_count, len(batch_list)))
            else:
                # 权重>1时，重复采样
                sampled_batches = random.choices(batch_list, k=sample_count)
            
            for i_batch, sampled_batch in sampled_batches:
                cls_batch_pool.append((dataset_name, num_classes, i_batch, sampled_batch))
        
        # 随机打乱采样池
        random.shuffle(cls_batch_pool)
        logging.info(f"Total classification batches after weighted sampling: {len(cls_batch_pool)}")
        
        # 训练加权采样后的批次
        for dataset_name, num_classes, i_batch, sampled_batch in tqdm(cls_batch_pool, total=len(cls_batch_pool), desc="Cls-Training"):
                    
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.to(device=device), label_batch.to(device=device)

                if args.prompt:
                    position_prompt = torch.stack(sampled_batch['position_prompt']).permute([1, 0]).float().to(device=device)
                    task_prompt = torch.stack(sampled_batch['task_prompt']).permute([1, 0]).float().to(device=device)
                    type_prompt = torch.stack(sampled_batch['type_prompt']).permute([1, 0]).float().to(device=device)
                    nature_prompt = torch.stack(sampled_batch['nature_prompt']).permute([1, 0]).float().to(device=device)
                    
                    # 直接调用对应数据集和分类数的decoder
                    x_cls = model((image_batch, position_prompt, task_prompt, type_prompt, nature_prompt),
                                  use_dataset_specific=True, dataset_name=dataset_name, task_type='cls', num_classes=num_classes)
                else:
                    x_cls = model(image_batch, use_dataset_specific=True, dataset_name=dataset_name, task_type='cls', num_classes=num_classes)

                # 计算分类损失
                if num_classes == 2:
                    loss = cls_ce_loss_2way(x_cls, label_batch[:].long())
                elif num_classes == 4:
                    loss = cls_ce_loss_4way(x_cls, label_batch[:].long())
                else:
                    raise ValueError(f"Unsupported num_classes: {num_classes}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                # lr_ = lr_scheduler.step()

                cls_iter_num = cls_iter_num + 1
                global_iter_num = global_iter_num + 1

                writer.add_scalar('info/lr_cls', lr_, cls_iter_num)
                writer.add_scalar(f'info/cls_loss_{dataset_name}', loss, cls_iter_num)
                writer.add_scalar('info/cls_loss_total', loss, cls_iter_num)

                if global_iter_num % 50 == 0:  # 减少日志频率
                    logging.info('Dataset: %s, global iteration %d and cls iteration %d : loss : %f' %
                                 (dataset_name, global_iter_num, cls_iter_num, loss.item()))

        
        # ========== 验证和模型保存逻辑 ==========

        
        dist.barrier()

        if int(os.environ["LOCAL_RANK"]) == 0:

            save_dict = {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'epoch': epoch_num}
            save_latest_path = os.path.join(snapshot_path, 'latest_{}.pth'.format(epoch_num))
            if os.path.exists(os.path.join(snapshot_path, 'latest_{}.pth'.format(epoch_num-1))):
                os.remove(os.path.join(snapshot_path, 'latest_{}.pth'.format(epoch_num-1)))
                os.remove(os.path.join(snapshot_path, 'latest.pth'))
            torch.save(save_dict, save_latest_path)
            os.system('ln -s ' + os.path.abspath(save_latest_path) + ' ' + os.path.join(snapshot_path, 'latest.pth'))
            # Only run validation every 2 epochs to speed up training
            if epoch_num % 2 != 0 and epoch_num != max_epoch - 1:  # Skip validation except every 2 epochs and last epoch
                logging.info(f"Skipping validation for epoch {epoch_num} to speed up training")
                model.train()
                continue  # Skip to next epoch
            else:
                # Continue with validation for every 2nd epoch or last epoch
                logging.info(f"Running validation for epoch {epoch_num}")
                model.eval()
                total_performance = 0.0

                seg_val_set = [
                    "private_Thyroid",
                    "private_Kidney",
                    "private_Fetal_Head",
                    "private_Cardiac",
                    "private_Breast_luminal",
                    "private_Breast",
                    ]
                seg_avg_performance = 0.0

                for dataset_name in seg_val_set:
                    num_classes = 2
                    db_val = USdatasetSeg(
                        base_dir=os.path.join(args.root_path,
                        "segmentation", dataset_name),
                        split="test",
                        list_dir=os.path.join(args.root_path, "segmentation", dataset_name),
                        transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
                        prompt=args.prompt
                    )
                    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=16)
                    logging.info("{} val iterations per epoch".format(len(val_loader)))

                    metric_list = 0.0
                    count_matrix = np.ones((len(db_val), num_classes-1))
                    for i_batch, sampled_batch in tqdm(enumerate(val_loader)):
                        image, label = sampled_batch["image"], sampled_batch["label"]
                        if args.prompt:
                            position_prompt = torch.stack(sampled_batch['position_prompt']).permute([1, 0]).float()
                            task_prompt = torch.tensor([[0]*position_prompt.shape[0], [1]*position_prompt.shape[0]]).permute([1, 0]).float()
                            type_prompt = torch.stack(sampled_batch['type_prompt']).permute([1, 0]).float()
                            nature_prompt = torch.stack(sampled_batch['nature_prompt']).permute([1, 0]).float()
        
                            metric_i = omni_seg_test_decoders(image, label, model,
                                                    classes=num_classes,
                                                    prompt=args.prompt,
                                                    type_prompt=type_prompt,
                                                    nature_prompt=nature_prompt,
                                                    position_prompt=position_prompt,
                                                    task_prompt=task_prompt,
                                                    dataset_name=dataset_name)
                        else:
                            metric_i = omni_seg_test_decoders(image, label, model,
                                                    classes=num_classes, dataset_name=dataset_name)

                        # 实际上这里的metric_i是多类别分割的结果，这个任务只有二类分割，只对一个标签做处理
                        for sample_index in range(len(metric_i[0][0])):
                            if not metric_i[0][1][sample_index]:
                                count_matrix[i_batch*batch_size+sample_index, 0] = 0
                        metric_i = metric_i[0][0]
                        metric_list += np.array(metric_i).sum()
                        print(len(metric_i))

                    metric_list = metric_list / (count_matrix.sum(axis=0) + 1e-6)
                    performance = np.mean(metric_list, axis=0)

                    writer.add_scalar('info/val_seg_metric_{}'.format(dataset_name), performance, epoch_num)

                    seg_avg_performance += performance

                seg_avg_performance = seg_avg_performance / (len(seg_val_set)+1e-6)
                total_performance += seg_avg_performance
                writer.add_scalar('info/val_metric_seg_Total', seg_avg_performance, epoch_num)
                
                cls_val_set = [
                    "private_Liver",
                    "private_Breast_luminal",
                    "private_Breast",
                    "private_Appendix",
                    ]
                cls_avg_performance = 0.0

                for dataset_name in cls_val_set:
                    if dataset_name == "private_Breast_luminal":
                        num_classes = 4
                    else:
                        num_classes = 2
                    db_val = USdatasetCls(
                        base_dir=os.path.join(args.root_path, "classification", dataset_name),
                        split="test",
                        list_dir=os.path.join(args.root_path, "classification", dataset_name),
                        transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
                        prompt=args.prompt
                    )

                    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=16)
                    logging.info("{} val iterations per epoch".format(len(val_loader)))
                    model.eval()

                    label_list = []
                    prediction_prob_list = []
                    for i_batch, sampled_batch in tqdm(enumerate(val_loader)):
                        image, label = sampled_batch["image"], sampled_batch["label"]
                        if args.prompt:
                            position_prompt = torch.stack(sampled_batch['position_prompt']).permute([1, 0]).float()
                            # task_prompt = torch.stack([[0]*position_prompt.shape[0], [1]*position_prompt.shape[0]]).permute([1, 0]).float()
                            task_prompt = torch.tensor([[0]*position_prompt.shape[0], [1]*position_prompt.shape[0]]).permute([1, 0]).float()
                            type_prompt = torch.stack(sampled_batch['type_prompt']).permute([1, 0]).float()
                            nature_prompt = torch.stack(sampled_batch['nature_prompt']).permute([1, 0]).float()
                            with torch.no_grad():
                                logits = model((image.cuda(), position_prompt.cuda(), task_prompt.cuda(),
                                            type_prompt.cuda(), nature_prompt.cuda()), use_dataset_specific=True, dataset_name=dataset_name, task_type='cls', num_classes=num_classes)
                        else:
                            with torch.no_grad():
                                logits = model(image.cuda(), use_dataset_specific=True, dataset_name=dataset_name, task_type='cls', num_classes=num_classes)


                        # if num_classes == 4:
                        #     logits = output[2]
                        # else:
                        #     logits = output[1]

                        output_prob = torch.softmax(logits, dim=1).data.cpu().numpy()

                        label_list.append(label.numpy())
                        prediction_prob_list.append(output_prob)

                    # label_list = np.expand_dims(np.concatenate(
                    #     (np.array(label_list[:-1]).flatten(), np.array(label_list[-1]).flatten())), axis=1).astype('uint8')
                    # label_list_OneHot = np.eye(num_classes)[label_list].squeeze(1)
                    # performance = roc_auc_score(label_list_OneHot, np.concatenate(
                    #     (np.array(prediction_prob_list[:-1]).reshape(-1, 2), prediction_prob_list[-1])), multi_class='ovo')
                    


                    label_list = np.expand_dims(np.concatenate(
                        (np.array(label_list[:-1]).flatten(), np.array(label_list[-1]).flatten())), axis=1).astype('uint8')
                    label_list_OneHot = np.eye(num_classes)[label_list].squeeze(1)
                    
                    prediction_probs_reshaped = np.array(prediction_prob_list[:-1]).reshape(-1, num_classes)
                    all_prediction_probs = np.concatenate((prediction_probs_reshaped, prediction_prob_list[-1]))

                    performance = roc_auc_score(label_list_OneHot, all_prediction_probs, multi_class='ovo')

                    writer.add_scalar('info/val_cls_metric_{}'.format(dataset_name), performance, epoch_num)

                    cls_avg_performance += performance

                cls_avg_performance = cls_avg_performance / (len(cls_val_set)+1e-6)
                total_performance += cls_avg_performance
                writer.add_scalar('info/val_metric_cls_Total', cls_avg_performance, epoch_num)

                TotalAvgPerformance = total_performance/2

                logging.info('This epoch %d Validation performance: %f' % (epoch_num, TotalAvgPerformance))
                logging.info('But the best epoch is: %d and performance: %f' % (best_epoch, best_performance))
                writer.add_scalar('info/val_metric_TotalMean', TotalAvgPerformance, epoch_num)
                # 替换原来的模型保存逻辑
                if TotalAvgPerformance >= best_performance or len(best_models) < max_saved_models:
                    # 创建新的模型保存路径
                    save_model_path = os.path.join(snapshot_path, 'best_model_{}_{}.pth'.format(
                        epoch_num, round(TotalAvgPerformance, 4)))
                    
                    # 保存当前模型
                    torch.save(model.state_dict(), save_model_path)
                    logging.info("save model to {}".format(save_model_path))
                    
                    # 添加到最佳模型列表
                    best_models.append((epoch_num, TotalAvgPerformance, save_model_path))
                    
                    # 按性能排序（降序）
                    best_models.sort(key=lambda x: x[1], reverse=True)
                    
                    # 如果超过最大保存数量，删除性能最差的模型
                    while len(best_models) > max_saved_models:
                        worst_model = best_models.pop()  # 移除最后一个（性能最差的）
                        if os.path.exists(worst_model[2]):
                            os.remove(worst_model[2])
                            logging.info("Removed model: {}".format(worst_model[2]))
                    
                    # 更新最佳性能和epoch
                    if TotalAvgPerformance > best_performance:
                        best_epoch = epoch_num
                        best_performance = TotalAvgPerformance
                        logging.info('New best validation TotalAvgPerformance: %f' % (TotalAvgPerformance))
                        
                        # 更新best_model.pth软链接指向当前最佳模型
                        best_model_link = os.path.join(snapshot_path, 'best_model.pth')
                        if os.path.exists(best_model_link) or os.path.islink(best_model_link):
                            os.remove(best_model_link)
                        os.system('ln -s ' + os.path.abspath(save_model_path) + ' ' + best_model_link)
                    
                    # 打印当前保存的所有最佳模型
                    logging.info("Current top {} models:".format(len(best_models)))
                    for i, (epoch, perf, path) in enumerate(best_models):
                        logging.info("  Rank {}: Epoch {} - Performance {:.4f} - {}".format(
                            i+1, epoch, perf, os.path.basename(path)))
                # Save model every 20 epochs
                if epoch_num % 20 == 0 and epoch_num > 0:
                    # Create a path for the periodic save
                    periodic_save_path = os.path.join(snapshot_path, 'epoch{}_performance_{}.pth'.format(
                        epoch_num, round(TotalAvgPerformance, 4)))
                    
                    # Save the model
                    torch.save(model.state_dict(), periodic_save_path)
                    logging.info("Saved periodic checkpoint at epoch {} to {}".format(
                        epoch_num, periodic_save_path))
                model.train()

        if TotalAvgPerformance > best_performance:
            best_performance = TotalAvgPerformance
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            logging.info(f"Early stopping triggered after {no_improve_count} epochs without improvement")
            break

    writer.close()
    return "Training Finished!"


def load_encoder_weights_and_freeze(model, resume_path):
    """
    仅加载encoder部分的权重并冻结encoder参数
    
    Args:
        model: 要加载权重的模型
        resume_path: 预训练模型路径
        
    Returns:
        resume_epoch: 恢复的epoch数
    """
    if resume_path is None:
        logging.info("No resume path provided, starting training from scratch")
        return 0
        
    logging.info(f"Loading encoder weights from: {resume_path}")
    
    try:
        checkpoint = torch.load(resume_path, map_location='cpu')
        
        # 提取预训练权重
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint
            
        # 获取当前模型的状态字典
        model_dict = model.state_dict()
        
        # 筛选出encoder相关的权重
        encoder_dict = {}
        # 更精确的encoder/decoder区分
        encoder_keywords = ['patch_embed', 'layers.', 'absolute_pos_embed', 'pos_drop','norm']
        decoder_keywords = ['seg_decoders', 'cls_decoders', 'seg_heads', 'cls_heads', 
                       'seg_skip_connections', 'cls_skip_connections', 'norm_task_seg', 'norm_task_cls']
        
        for k, v in pretrained_dict.items():
            # 检查是否是encoder相关的参数
            is_encoder = any(enc_key in k for enc_key in encoder_keywords)
            is_decoder = any(dec_key in k for dec_key in decoder_keywords)
            
            # 只加载encoder部分，排除decoder部分
            if is_encoder and not is_decoder:
                if k in model_dict and model_dict[k].shape == v.shape:
                    encoder_dict[k] = v
                    logging.info(f"Loading encoder weight: {k}, shape: {v.shape}")
                else:
                    if k not in model_dict:
                        logging.warning(f"Encoder weight not found in current model: {k}")
                    else:
                        logging.warning(f"Shape mismatch for {k}: pretrained {v.shape} vs model {model_dict[k].shape}")
        
        # 更新模型字典
        model_dict.update(encoder_dict)
        model.load_state_dict(model_dict)
        
        logging.info(f"Successfully loaded {len(encoder_dict)} encoder parameters from pretrained model")
        
        # 冻结encoder部分的权重
        frozen_params = 0
        trainable_params = 0
        frozen_param_names = []
        trainable_param_names = []
        
        for name, param in model.named_parameters():
            # 检查是否是encoder参数
            is_encoder = any(enc_key in name for enc_key in encoder_keywords)
            is_decoder = any(dec_key in name for dec_key in decoder_keywords)
            
            if is_encoder and not is_decoder:
                param.requires_grad = False
                frozen_params += param.numel()
                frozen_param_names.append(name)
            else:
                param.requires_grad = True
                trainable_params += param.numel()
                trainable_param_names.append(name)
                
        logging.info(f"Frozen {frozen_params:,} encoder parameters")
        logging.info(f"Trainable {trainable_params:,} decoder parameters")
        logging.info(f"Frozen parameter ratio: {frozen_params/(frozen_params+trainable_params)*100:.2f}%")
        
        # 获取恢复的epoch（如果有的话）
        if 'epoch' in checkpoint:
            resume_epoch = checkpoint['epoch']
            logging.info(f"Resuming from epoch {resume_epoch}")
        else:
            resume_epoch = 0
            logging.info("Starting fresh training with pretrained encoder")
            
        # 注意：不加载optimizer状态，因为我们改变了可训练参数的结构
        logging.info("Note: Optimizer state not loaded due to changed trainable parameter structure")
        
        return resume_epoch
        
    except Exception as e:
        logging.error(f"Error loading encoder weights: {e}")
        logging.info("Starting training from scratch")
        return 0

def get_dynamic_loss_weights(epoch, total_epochs):
    """根据训练进度动态调整损失权重"""
    progress = epoch / total_epochs
    if progress < 0.3:
        return 0.5, 0.5  # 早期平衡
    elif progress < 0.7:
        return 0.3, 0.7  # 中期偏重dice
    else:
        return 0.2, 0.8  # 后期更重dice

def weighted_batch_sampling(dataloaders, weights):
    """使用加权采样而不是随机跳过"""
    all_batches = []
    for dataset_name, info in dataloaders.items():
        weight = weights.get(dataset_name, 1.0)
        for batch in info['loader']:
            all_batches.append((dataset_name, batch, weight))
    
    # 按权重采样
    weights_list = [w for _, _, w in all_batches]
    return random.choices(all_batches, weights=weights_list, k=len(all_batches))