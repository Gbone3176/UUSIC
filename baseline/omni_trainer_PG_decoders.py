
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


from utils import DiceLoss
from datasets.dataset_aug_norm import USdatasetCls, USdatasetSeg
from datasets.dataset_aug_norm import RandomGenerator_Cls, RandomGenerator_Seg, CenterCropGenerator
from datasets.omni_dataset_decoders import WeightedRandomSamplerDDP
from datasets.omni_dataset_decoders import USdatasetOmni_cls_decoders, USdatasetOmni_seg_decoders
from sklearn.metrics import roc_auc_score
from utils import omni_seg_test


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
    seg_dataset_weights = {
        'private_Breast': 0.25,
        'private_Breast_luminal': 0.25,
        'private_Cardiac': 4,
        'private_Fetal_Head': 4,
        'private_Kidney': 2,
        'private_Thyroid': 4
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
    cls_dataset_weights = {
        'private_Appendix': 4,
        'private_Breast': 4,
        'private_Breast_luminal': 1,
        'private_Liver': 2
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

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01, betas=(0.9, 0.999))

    resume_epoch = 0
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location='cpu')['model'])
        optimizer.load_state_dict(torch.load(args.resume, map_location='cpu')['optimizer'])
        resume_epoch = torch.load(args.resume, map_location='cpu')['epoch']

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

    for epoch_num in iterator:
        logging.info("\n epoch: {}".format(epoch_num))
        
        # 为每个数据集的sampler设置epoch
        for dataset_name, info in seg_dataloaders.items():
            info['sampler'].set_epoch(epoch_num)
        for dataset_name, info in cls_dataloaders.items():
            info['sampler'].set_epoch(epoch_num)

        # ========== 分割任务训练 - 按数据集逐个训练 ==========
        logging.info("Training segmentation tasks...")
        for dataset_name, dataset_info in seg_dataloaders.items():
            dataloader = dataset_info['loader']
            weight = dataset_info['weight']
            
            logging.info(f"Training segmentation on {dataset_name} with weight {weight}")
            
            # 根据权重决定是否跳过某些批次（权重采样的替代方案）
            skip_prob = max(0.0, 1.0 - weight)
            
            for i_batch, sampled_batch in tqdm(enumerate(dataloader), desc=f"Seg-{dataset_name}"):
                # 根据权重随机跳过某些批次
                if random.random() < skip_prob:
                    continue
                   
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
                loss = 0.28 * loss_ce + 0.72 * loss_dice

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 修改学习率衰减策略：当进度超过70%时保持恒定
                progress = 1.0 - global_iter_num / max_iterations
                if progress >= 0.3:
                    lr_ = base_lr * (progress ** 0.9)
                else:
                    lr_ = base_lr * (0.3 ** 0.9)  # 保持在30%进度时的学习率
                    
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                seg_iter_num = seg_iter_num + 1
                global_iter_num = global_iter_num + 1

                writer.add_scalar('info/lr_seg', lr_, seg_iter_num)
                writer.add_scalar(f'info/seg_loss_{dataset_name}', loss, seg_iter_num)
                writer.add_scalar('info/seg_loss_total', loss, seg_iter_num)

                if global_iter_num % 50 == 0:  # 减少日志频率
                    logging.info('Dataset: %s, global iteration %d and seg iteration %d : loss : %f' %
                                 (dataset_name, global_iter_num, seg_iter_num, loss.item()))

        # ========== 分类任务训练 - 按数据集逐个训练 ==========
        logging.info("Training classification tasks...")
        for dataset_name, dataset_info in cls_dataloaders.items():
            dataloader = dataset_info['loader']
            weight = dataset_info['weight']
            num_classes = dataset_info['num_classes']
            
            logging.info(f"Training classification on {dataset_name} with {num_classes} classes and weight {weight}")
            
            # 根据权重决定是否跳过某些批次
            skip_prob = max(0.0, 1.0 - weight)
            
            for i_batch, sampled_batch in tqdm(enumerate(dataloader), desc=f"Cls-{dataset_name}"):
                # 根据权重随机跳过某些批次
                if random.random() < skip_prob:
                    continue
                    
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

                progress = 1.0 - global_iter_num / max_iterations
                if progress >= 0.3:
                    lr_ = base_lr * (progress ** 0.9)
                else:
                    lr_ = base_lr * (0.3 ** 0.9)  # 保持在30%进度时的学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

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
        
                            metric_i = omni_seg_test(image, label, model,
                                                    classes=num_classes,
                                                    prompt=args.prompt,
                                                    type_prompt=type_prompt,
                                                    nature_prompt=nature_prompt,
                                                    position_prompt=position_prompt,
                                                    task_prompt=task_prompt,
                                                    dataset_name=dataset_name)
                        else:
                            metric_i = omni_seg_test(image, label, model,
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

    writer.close()
    return "Training Finished!"