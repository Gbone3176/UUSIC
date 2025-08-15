
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
import torch.nn.functional as F


from utils import DiceLoss
from datasets.dataset_aug_norm import USdatasetCls, USdatasetSeg
from datasets.dataset_aug_norm_mc import RandomGenerator_Cls, RandomGenerator_Seg, CenterCropGenerator
from datasets.omni_dataset import WeightedRandomSamplerDDP
from datasets.omni_dataset import USdatasetOmni_cls_mix, USdatasetOmni_seg
from sklearn.metrics import roc_auc_score, accuracy_score
from utils import omni_seg_test_TU
from mixup_cutmix_utils import (
    apply_mixup_cutmix, 
    compute_mixup_loss, 
    log_augmentation_info
)


def freeze_all_but_classification_head(model):
    """
    冻结除 MultiTaskClassifier 以外的所有参数；整个分类器模块参与训练。
    返回：可训练参数列表（便于构建优化器）
    """
    # 兼容 DDP
    net = model.module if hasattr(model, "module") else model

    # 1) 全部冻结
    for p in net.parameters():
        p.requires_grad = False

    # 2) 解冻整个 MultiTaskClassifier（包括 norm, fc1, fc2, head2, head4）
    trainable_params = []
    for n, p in net.classifier_head.named_parameters():
        p.requires_grad = True
        trainable_params.append(p)

    # 3) 切换模式：全网 eval，分类器 train（使其 BN/Dropout 生效）
    model.eval()
    net.classifier_head.train()

    # 4) 打印统计
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[Freeze] trainable params: {trainable:,} / {total:,} ({trainable/total*100:.4f}%) "
          f"| MultiTaskClassifier: norm+fc1+fc2+head2+head4")

    return trainable_params


def classifier_trainable_params(model):
    """Iterator for params with requires_grad=True (safe for optimizer)."""
    return (p for p in model.parameters() if p.requires_grad)

def to_chw(x):
    return x.permute(0, 3, 1, 2).contiguous() if x.dim()==4 and x.size(-1) in (1, 3) else x

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

    db_train_seg = USdatasetOmni_seg(base_dir=args.root_path, split="train", transform=transforms.Compose(
        [RandomGenerator_Seg(output_size=[args.img_size, args.img_size])]), prompt=args.prompt)

    seg_datasets_weight = {
        'private_Breast': (105, 3),         # 对应weight_base[1] = 3
        'BUS-BRA': (1312, 0.25),            # 对应weight_base[7] = 0.25
        'BUSI': (452, 1),                   # 对应weight_base[2] = 1
        'BUSIS': (393, 1),                  # 对应weight_base[5] = 1
        'CAMUS': (2086, 0.25),              # 对应weight_base[6] = 0.25
        'DDTI': (7952, 0.25),               # 对应weight_base[0] = 0.25 (Thyroid相关)
        'Fetal_HC': (699, 0.5),             # 对应weight_base[12] = 0.5
        'KidneyUS': (340, 1),               # 对应weight_base[8] = 1
        'private_Breast_luminal': (165, 3), # 对应weight_base[10] = 2
        'private_Cardiac': (53, 2),         # 对应weight_base[11] = 4
        'private_Fetal_Head': (84, 1),      # 对应weight_base[4] = 4
        'private_Kidney': (46, 4),          # 对应weight_base[3] = 4
        'private_Thyroid': (299, 4)         # 对应weight_base[9] = 4
    }
    
    sample_weight_seq = [[seg_datasets_weight[seg_subset_name][1]] *
                         element for (seg_subset_name, element) in db_train_seg.subset_len]
    sample_weight_seq = [element for sublist in sample_weight_seq for element in sublist]

    weighted_sampler_seg = WeightedRandomSamplerDDP(
        data_set=db_train_seg,
        weights=sample_weight_seq,
        num_replicas=world_size,
        rank=rank,
        num_samples=len(db_train_seg),
        replacement=True
    )
    trainloader_seg = DataLoader(db_train_seg,
                                 batch_size=batch_size,
                                 num_workers=32,
                                 pin_memory=True,
                                 worker_init_fn=worker_init_fn,
                                 sampler=weighted_sampler_seg
                                 )

    # ====== 按数据集分别构建分类数据加载器 ======
    # 每个数据集单独构建DataLoader，避免不同病灶类型混合
    cls_datasets = {
        'Appendix': (981, 1, 2),           # (样本数, 权重, 类别数)
        'BUS-BRA': (1312, 0.25, 2),
        'BUSI': (452, 1, 2),
        'Fatty-Liver': (385, 1, 2),
        'private_Appendix': (46, 3, 2),
        'private_Breast': (105, 4, 2),
        'private_Liver': (72, 4, 2),
        'private_Breast_luminal': (165, 3, 4)
    }
    
    # 为每个数据集创建单独的DataLoader
    trainloaders_cls = {}
    weighted_samplers_cls = {}
    
    for dataset_name, (sample_count, weight, num_classes) in cls_datasets.items():
        # 创建只包含单个数据集的数据集对象
        db_train_cls_single = USdatasetOmni_cls_mix(
            base_dir=args.root_path, 
            split="train", 
            transform=transforms.Compose([RandomGenerator_Cls(output_size=[args.img_size, args.img_size])]), 
            prompt=args.prompt,
            dataset_name=dataset_name
            )
        
        if len(db_train_cls_single) > 0:
            # 创建权重采样器
            sample_weight_seq = [weight] * len(db_train_cls_single)
            
            weighted_sampler = WeightedRandomSamplerDDP(
                data_set=db_train_cls_single,
                weights=sample_weight_seq,
                num_replicas=world_size,
                rank=rank,
                num_samples=len(db_train_cls_single),
                replacement=True
            )
            
            # 创建DataLoader
            trainloader = DataLoader(
                db_train_cls_single,
                batch_size=batch_size,
                num_workers=32,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                sampler=weighted_sampler
            )
            
            trainloaders_cls[dataset_name] = {
                'loader': trainloader,
                'sampler': weighted_sampler,
                'num_classes': num_classes,
                'weight': weight
            }
    
    # 计算总迭代次数
    total_cls_iterations = sum(len(info['loader']) for info in trainloaders_cls.values())

    model = model.to(device=device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)

    # 只训练分类器：先冻结，再用仅包含可训练参数的优化器
    # freeze_all_but_classification_head(model)

    model.train()

    seg_ce_loss = CrossEntropyLoss()
    seg_dice_loss = DiceLoss()

    weight = torch.tensor([6.94,10.72, 1.52, 9.44], dtype=torch.float32).to(device)
    cls_ce_loss_2way = CrossEntropyLoss()
    cls_ce_loss_4way = CrossEntropyLoss(weight=weight)

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
    total_iterations = (len(trainloader_seg) + total_cls_iterations)
    max_iterations = args.max_epochs * total_iterations
    logging.info("{} batch size. {} iterations per epoch. {} max iterations ".format(
        batch_size, total_iterations, max_iterations))
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
        weighted_sampler_seg.set_epoch(epoch_num)
        # 设置分类数据采样器的epoch
        for dataset_name, info in trainloaders_cls.items():
            info['sampler'].set_epoch(epoch_num)

        for i_batch, sampled_batch in tqdm(enumerate(trainloader_seg), total=len(trainloader_seg)):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            image_batch = to_chw(image_batch).to(device=device)

            image_batch, label_batch = image_batch.to(device=device), label_batch.to(device=device)
            if args.prompt:
                position_prompt = torch.stack(sampled_batch['position_prompt']).permute([1, 0]).float().to(device=device)
                task_prompt = torch.stack(sampled_batch['task_prompt']).permute([1, 0]).float().to(device=device)
                type_prompt = torch.stack(sampled_batch['type_prompt']).permute([1, 0]).float().to(device=device)
                nature_prompt = torch.stack(sampled_batch['nature_prompt']).permute([1, 0]).float().to(device=device) 
                
                (x_seg, _, _) = model(image_batch, position_prompt, task_prompt, type_prompt, nature_prompt)
            else:
                (x_seg, _, _) = model(image_batch)

            loss_ce = seg_ce_loss(x_seg, label_batch[:].long())
            loss_dice = seg_dice_loss(x_seg, label_batch, softmax=True)
            loss = 0.28 * loss_ce + 0.72 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 修改学习率衰减策略：当进度超过70%时保持恒定
            progress = 1.0 - global_iter_num / max_iterations
            if progress >= 0.2:
                lr_ = base_lr * (progress ** 0.9)
            else:
                lr_ = base_lr * (0.2 ** 0.9)  # 保持在20%进度时的学习率
                
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            seg_iter_num = seg_iter_num + 1
            global_iter_num = global_iter_num + 1

            writer.add_scalar('info/lr_seg', lr_, seg_iter_num)
            writer.add_scalar('info/seg_loss', loss, seg_iter_num)

            logging.info('global iteration %d and seg iteration %d : loss : %f' %
                         (global_iter_num, seg_iter_num, loss.item()))

        
        # ====== 按数据集分别训练分类任务 ======
        for dataset_name, info in trainloaders_cls.items():
            trainloader = info['loader']
            num_classes = info['num_classes']
            
            # 选择对应的损失函数
            if num_classes == 2:
                cls_loss_fn = cls_ce_loss_2way
                output_idx = 1  # x_cls_2
            else:  # num_classes == 4
                cls_loss_fn = cls_ce_loss_4way
                output_idx = 2  # x_cls_4
            
            for i_batch, sampled_batch in tqdm(enumerate(trainloader), total=len(trainloader), 
                                               desc=f"Training {dataset_name}"):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                
                image_batch = to_chw(image_batch).to(device=device)
                label_batch = label_batch.to(device=device)
                
                # ====== 应用 Mixup/CutMix 数据增强 ======
                # 每个batch中的样本都是同一数据集，类别数一致，可以安全使用mixup/cutmix
                image_batch, label_batch_a, label_batch_b, lam, augmentation_type = apply_mixup_cutmix(
                    image_batch, label_batch, torch.full((image_batch.size(0),), num_classes, device=device), args, device
                )
                
                if args.prompt:
                    position_prompt = torch.stack(sampled_batch['position_prompt']).permute([1, 0]).float().to(device=device)
                    task_prompt = torch.stack(sampled_batch['task_prompt']).permute([1, 0]).float().to(device=device)
                    type_prompt = torch.stack(sampled_batch['type_prompt']).permute([1, 0]).float().to(device=device)
                    nature_prompt = torch.stack(sampled_batch['nature_prompt']).permute([1, 0]).float().to(device=device)
                    model_output = model(image_batch, position_prompt, task_prompt, type_prompt, nature_prompt)
                else:
                    model_output = model(image_batch)
                
                # 获取对应的分类输出
                x_cls = model_output[output_idx]

                # 计算损失
                if augmentation_type > 0:  # 使用了Mixup或CutMix
                    loss_ce = compute_mixup_loss(
                        x_cls, label_batch_a, label_batch_b, lam, 
                        cls_loss_fn, num_classes, f"{dataset_name}_classification"
                    )
                else:  # 未使用数据增强
                    loss_ce = cls_loss_fn(x_cls, label_batch[:].long())

                optimizer.zero_grad()
                loss_ce.backward()
                optimizer.step()

                progress = 1.0 - global_iter_num / max_iterations
                if progress >= 0.3:
                    lr_ = base_lr * (progress ** 0.9)
                else:
                    lr_ = base_lr * (0.3 ** 0.9)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                cls_iter_num = cls_iter_num + 1
                global_iter_num = global_iter_num + 1

                # 记录数据增强相关信息
                writer.add_scalar(f'info/lr_cls_{dataset_name}', lr_, cls_iter_num)
                writer.add_scalar(f'info/cls_loss_{dataset_name}', loss_ce, cls_iter_num)
                log_augmentation_info(writer, cls_iter_num, augmentation_type, lam, f"cls_{dataset_name}")

                logging.info('global iteration %d and %s iteration %d : loss : %f, aug_type: %d, lam: %.3f' %
                             (global_iter_num, dataset_name, cls_iter_num, loss_ce.item(), augmentation_type, lam))
        
        
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

            model.eval()
            total_performance = 0.0

            seg_val_set = [
                # "BUS-BRA",
                # "BUSIS",
                # "BUSI",
                # "CAMUS",
                # "DDTI",
                # "Fetal_HC",
                # "KidneyUS",
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
                    image = to_chw(image)
                    if args.prompt:
                        position_prompt = torch.stack(sampled_batch['position_prompt']).permute([1, 0]).float()
                        task_prompt = torch.tensor([[0]*position_prompt.shape[0], [1]*position_prompt.shape[0]]).permute([1, 0]).float()
                        type_prompt = torch.stack(sampled_batch['type_prompt']).permute([1, 0]).float()
                        nature_prompt = torch.stack(sampled_batch['nature_prompt']).permute([1, 0]).float()
                        metric_i = omni_seg_test_TU(image, label, model,
                                                 classes=num_classes,
                                                 prompt=args.prompt,
                                                 type_prompt=type_prompt,
                                                 nature_prompt=nature_prompt,
                                                 position_prompt=position_prompt,
                                                 task_prompt=task_prompt,
                                                 dataset_name=dataset_name
                                                 )
                    else:
                        metric_i = omni_seg_test_TU(image, label, model,
                                                 classes=num_classes, dataset_name=dataset_name)

                    # 实际上这里的metric_i是多类别分割的结果，这个任务只有二类分割，只对一个标签做处理
                    for sample_index in range(len(metric_i[0][0])):
                        if not metric_i[0][1][sample_index]:
                            count_matrix[i_batch*batch_size+sample_index, 0] = 0
                    metric_i = metric_i[0][0]
                    metric_list += np.array(metric_i).sum()
                    # print(len(metric_i))

                metric_list = metric_list / (count_matrix.sum(axis=0) + 1e-6)
                performance = np.mean(metric_list, axis=0)

                writer.add_scalar('info/val_seg_metric_{}'.format(dataset_name), performance, epoch_num)
                logging.info('This epoch %d %s seg Val performance(Dice): %f' % (epoch_num, dataset_name, performance))
                seg_avg_performance += performance

            seg_avg_performance = seg_avg_performance / (len(seg_val_set)+1e-6)
            total_performance += seg_avg_performance
            writer.add_scalar('info/val_metric_seg_Total', seg_avg_performance, epoch_num)
            
            cls_val_set = [
                # "Appendix",
                # "BUS-BRA",
                # "BUSI",
                # "Fatty-Liver",
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
                    image = to_chw(image)
                    if args.prompt:
                        position_prompt = torch.stack(sampled_batch['position_prompt']).permute([1, 0]).float()
                        # task_prompt = torch.stack([[0]*position_prompt.shape[0], [1]*position_prompt.shape[0]]).permute([1, 0]).float()
                        task_prompt = torch.tensor([[0]*position_prompt.shape[0], [1]*position_prompt.shape[0]]).permute([1, 0]).float()
                        type_prompt = torch.stack(sampled_batch['type_prompt']).permute([1, 0]).float()
                        nature_prompt = torch.stack(sampled_batch['nature_prompt']).permute([1, 0]).float()
                        with torch.no_grad():
                            output = model((image.cuda(), position_prompt.cuda(), task_prompt.cuda(),
                                           type_prompt.cuda(), nature_prompt.cuda()))
                    else:
                        with torch.no_grad():
                            output = model(image.cuda())


                    if num_classes == 4:
                        logits = output[2]
                    else:
                        logits = output[1]

                    output_prob = torch.softmax(logits, dim=1).data.cpu().numpy()

                    label_list.append(label.numpy())
                    prediction_prob_list.append(output_prob)

                label_list = np.expand_dims(np.concatenate(
                    (np.array(label_list[:-1]).flatten(), np.array(label_list[-1]).flatten())), axis=1).astype('uint8')
                label_list_OneHot = np.eye(num_classes)[label_list].squeeze(1)
                
                prediction_probs_reshaped = np.array(prediction_prob_list[:-1]).reshape(-1, num_classes)
                all_prediction_probs = np.concatenate((prediction_probs_reshaped, prediction_prob_list[-1]))

                # performance = roc_auc_score(label_list_OneHot, all_prediction_probs, multi_class='ovo')

                y_true = np.argmax(label_list_OneHot, axis=1)
                y_pred = np.argmax(all_prediction_probs, axis=1)
                performance = accuracy_score(y_true, y_pred)

                writer.add_scalar('info/val_cls_metric_{}'.format(dataset_name), performance, epoch_num)
                logging.info('This epoch %d %s cls Val performance(Acc): %f' % (epoch_num, dataset_name, performance))
                cls_avg_performance += performance

            cls_avg_performance = cls_avg_performance / (len(cls_val_set)+1e-6)
            total_performance += cls_avg_performance
            writer.add_scalar('info/val_metric_cls_Total', cls_avg_performance, epoch_num)
            


            TotalAvgPerformance = total_performance/2
            # TotalAvgPerformance = total_performance

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