
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
from datasets.omni_dataset import WeightedRandomSamplerDDP
from datasets.omni_dataset import USdatasetOmni_cls, USdatasetOmni_seg
from sklearn.metrics import roc_auc_score
from utils import omni_seg_test_TU

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

    db_train_cls = USdatasetOmni_cls(base_dir=args.root_path, split="train", transform=transforms.Compose(
        [RandomGenerator_Cls(output_size=[args.img_size, args.img_size])]), prompt=args.prompt)

    cls_datasets_weight = {
        'Appendix': (981, 1),                # 默认权重1（未在weight_base中明确对应）
        'BUS-BRA': (1312, 0.25),             # 对应weight_base[7] = 0.25 (大规模数据低权重)
        'BUSI': (452, 1),                    # 对应weight_base[2] = 1
        'Fatty-Liver': (385, 1),             # 接近weight_base[6]的350样本，权重=1
        'private_Appendix': (46, 2),         # 对应weight_base[3] = 4 (小样本高权重)
        'private_Breast': (105, 4),          # 对应weight_base[1] = 3
        'private_Breast_luminal': (165, 1),  # 对应weight_base[10] = 2
        'private_Liver': (72, 4)             # 接近weight_base[4]的84样本，权重=4
    }

    sample_weight_seq = [[cls_datasets_weight[cls_subset_name][1]] *element 
                         for (cls_subset_name, element) in db_train_cls.subset_len]
    sample_weight_seq = [element for sublist in sample_weight_seq for element in sublist]

    weighted_sampler_cls = WeightedRandomSamplerDDP(
        data_set=db_train_cls,
        weights=sample_weight_seq,
        num_replicas=world_size,
        rank=rank,
        num_samples=len(db_train_cls),
        replacement=True
    )
    trainloader_cls = DataLoader(db_train_cls,
                                 batch_size=batch_size,
                                 num_workers=32,
                                 pin_memory=True,
                                 worker_init_fn=worker_init_fn,
                                 sampler=weighted_sampler_cls
                                 )

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
    total_iterations = (len(trainloader_seg) + len(trainloader_cls))
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
        weighted_sampler_cls.set_epoch(epoch_num)

        for i_batch, sampled_batch in tqdm(enumerate(trainloader_seg), total=len(trainloader_seg)):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            image_batch = to_chw(image_batch).to(device=device)

            image_batch, label_batch = image_batch.to(device=device), label_batch.to(device=device)
            if args.prompt:
                position_prompt = torch.stack(sampled_batch['position_prompt']).permute([1, 0]).float().to(device=device)
                task_prompt = torch.stack(sampled_batch['task_prompt']).permute([1, 0]).float().to(device=device)
                type_prompt = torch.stack(sampled_batch['type_prompt']).permute([1, 0]).float().to(device=device)
                nature_prompt = torch.stack(sampled_batch['nature_prompt']).permute([1, 0]).float().to(device=device) 
                
                (x_seg, _, _) = model((image_batch, position_prompt, task_prompt, type_prompt, nature_prompt))
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

        
        for i_batch, sampled_batch in tqdm(enumerate(trainloader_cls), total=len(trainloader_cls)):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            
            num_classes_batch = sampled_batch['num_classes']
            image_batch = to_chw(image_batch).to(device=device)
            label_batch = label_batch.to(device=device)
            if args.prompt:
                position_prompt = torch.stack(sampled_batch['position_prompt']).permute([1, 0]).float().to(device=device)
                task_prompt = torch.stack(sampled_batch['task_prompt']).permute([1, 0]).float().to(device=device)
                type_prompt = torch.stack(sampled_batch['type_prompt']).permute([1, 0]).float().to(device=device)
                nature_prompt = torch.stack(sampled_batch['nature_prompt']).permute([1, 0]).float().to(device=device)
                (_, x_cls_2, x_cls_4) = model((image_batch, position_prompt, task_prompt, type_prompt, nature_prompt))
            else:
                (_, x_cls_2, x_cls_4) = model(image_batch)

            loss = 0.0
            
            mask_2_way = (num_classes_batch == 2)
            mask_4_way = (num_classes_batch == 4)


            if mask_2_way.any():
                outputs_2_way = x_cls_2[mask_2_way]
                labels_2_way = label_batch[mask_2_way]
                loss_ce_2 = cls_ce_loss_2way(outputs_2_way, labels_2_way[:].long())
                loss += loss_ce_2


            if mask_4_way.any():
                outputs_4_way = x_cls_4[mask_4_way]
                labels_4_way = label_batch[mask_4_way]
                loss_ce_4 = cls_ce_loss_4way(outputs_4_way, labels_4_way[:].long())
                loss += loss_ce_4

            # loss_ce = cls_ce_loss(x_cls, label_batch[:].long())
            # loss = loss_ce

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
            writer.add_scalar('info/cls_loss', loss, cls_iter_num)

            logging.info('global iteration %d and cls iteration %d : loss : %f' %
                         (global_iter_num, cls_iter_num, loss.item()))
        
        
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
                                                 task_prompt=task_prompt
                                                 )
                    else:
                        metric_i = omni_seg_test_TU(image, label, model,
                                                 classes=num_classes)

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

                performance = roc_auc_score(label_list_OneHot, all_prediction_probs, multi_class='ovo')

                writer.add_scalar('info/val_cls_metric_{}'.format(dataset_name), performance, epoch_num)

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