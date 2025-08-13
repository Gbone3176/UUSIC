import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from datasets.dataset_aug_norm_mc import CenterCropGenerator
from datasets.dataset_aug_norm import USdatasetClsFlexible, USdatasetSegFlexible
from datasets.omni_dataset import position_prompt_one_hot_dict, task_prompt_one_hot_dict, type_prompt_one_hot_dict, nature_prompt_one_hot_dict

POSITION_LEN = len(position_prompt_one_hot_dict)
TASK_LEN = len(task_prompt_one_hot_dict)
TYPE_LEN = len(type_prompt_one_hot_dict)
NAT_LEN = len(nature_prompt_one_hot_dict)


from utils import omni_seg_test_TU
from sklearn.metrics import accuracy_score

# === 使用 TransUNet ===
from networks.vit_seg_modeling_v2 import VisionTransformer, CONFIGS as VIT_CONFIGS


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='data/', help='root dir for data')
parser.add_argument('--output_dir', type=str, required=True, help='output dir')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per GPU for inference')
parser.add_argument('--img_size', type=int, default=224, help='input size (H=W)')
parser.add_argument('--is_saveout', action="store_true", help='save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction dir')
parser.add_argument('--deterministic', type=int, default=1, help='deterministic')
parser.add_argument('--seed', type=int, default=1234, help='random seed')

# —— TransUNet 相关 —— #
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                    choices=['ViT-B_16','ViT-B_32','ViT-L_16','ViT-L_32','ViT-H_14','R50-ViT-B_16','R50-ViT-L_16'])
parser.add_argument('--vit_pretrained', type=str, default=None,
                    help='.npz pretrained path (if None, use cfg.pretrained_path)')

# —— checkpoint —— #
parser.add_argument('--resume', type=str, required=True, help='model checkpoint (.pth) for inference')

parser.add_argument('--prompt', action='store_true', help='(deprecated here) using prompt for training')
parser.add_argument('--adapter_ft', action='store_true', help='(deprecated here) using adapter for fine-tuning')

args = parser.parse_args()


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_ddp():
    # torchrun 会自动设置这些环境变量
    if 'LOCAL_RANK' not in os.environ:
        # 允许单进程无 torchrun 运行
        return None, torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda', local_rank)
    return local_rank, device


# NHWC -> NCHW（如果需要）
def to_chw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous() if x.dim() == 4 and x.size(-1) in (1, 3) else x


@torch.no_grad()
def inference(args, model, device, test_save_path=None):
    import csv, time

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        result_csv = f"{args.output_dir}/result_{os.path.splitext(os.path.basename(args.resume))[0]}.csv"
        if not os.path.exists(result_csv):
            with open(result_csv, 'w', newline='') as f:
                csv.writer(f).writerow(['dataset', 'task', 'metric', 'time'])
    else:
        result_csv = None

    seg_test_set = [
        "private_Thyroid",
        "private_Kidney",
        "private_Fetal_Head",
        "private_Cardiac",
        "private_Breast_luminal",
        "private_Breast",
    ]
    cls_test_set = [
        "private_Liver",
        "private_Breast_luminal",
        "private_Breast",
        "private_Appendix",
    ]

    private_performence_seg = {}
    private_performence_cls = {}

    # ===== Segmentation =====
    for dataset_name in seg_test_set:
        num_classes = 2
        ds = USdatasetSegFlexible(
            base_dir=os.path.join(args.root_path, "segmentation", dataset_name),
            split=["test"],
            list_dir=os.path.join(args.root_path, "segmentation", dataset_name),
            transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt
        )
        sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if dist.is_initialized() else None
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False if sampler else False,
                            num_workers=12, pin_memory=True, sampler=sampler)

        if is_main_process():
            logging.info("%s: %d samples", dataset_name, len(ds))

        model.eval()
        metric_list = 0.0
        count_matrix = np.ones((len(ds), num_classes - 1))
        # 非主进程关闭 tqdm
        iterator = tqdm(enumerate(loader), total=len(loader), disable=not is_main_process())

        base_idx = 0
        for i_batch, sampled_batch in iterator:
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name']
            case_name = case_name[0] if isinstance(case_name, (list, tuple)) else str(case_name)
            image = to_chw(image).to(device, non_blocking=True)
            if args.prompt:
                position_prompt = torch.tensor(np.array(sampled_batch['position_prompt'])).permute([1, 0]).float()
                task_prompt = torch.tensor(np.array([[1], [0]])).permute([1, 0]).float()
                type_prompt = torch.tensor(np.array(sampled_batch['type_prompt'])).permute([1, 0]).float()
                nature_prompt = torch.tensor(np.array(sampled_batch['nature_prompt'])).permute([1, 0]).float()
                metric_i = omni_seg_test_TU(image, label, model,
                                         classes=num_classes,
                                         test_save_path=test_save_path,
                                         case=case_name,
                                         prompt=args.prompt,
                                         type_prompt=type_prompt,
                                         nature_prompt=nature_prompt,
                                         position_prompt=position_prompt,
                                         task_prompt=task_prompt,
                                         dataset_name=dataset_name
                                         )
            else:
                metric_i = omni_seg_test_TU(image, label, model,
                                         classes=num_classes,
                                         test_save_path=test_save_path,
                                         case=case_name, dataset_name=dataset_name)
            
            zero_label_flag = False
            for i in range(1, num_classes):
                if not metric_i[i - 1][1]:
                    idx = base_idx + i_batch  # 简单线性索引
                    if idx < len(ds):
                        count_matrix[idx, i - 1] = 0
                    zero_label_flag = True

            metric_vals = metric_i[0][0]
            metric_list += np.array(metric_vals).sum()

            if is_main_process():
                iterator.set_postfix(mean_dice=float(np.mean(metric_vals)))

        # 同步 reduce
        if dist.is_initialized():
            tensor_ml = torch.tensor(metric_list, device=device)
            tensor_cm = torch.tensor(count_matrix, device=device)
            dist.all_reduce(tensor_ml, op=dist.ReduceOp.SUM)
            dist.all_reduce(tensor_cm, op=dist.ReduceOp.SUM)
            metric_list = tensor_ml.item()
            count_matrix = tensor_cm.cpu().numpy()

        metric_list = metric_list / (count_matrix.sum(axis=0) + 1e-6)
        performance = float(np.mean(metric_list, axis=0))

        if is_main_process():
            logging.info('[SEG] %s mean_dice: %.4f', dataset_name, performance)
            if "private_" in dataset_name:
                private_performence_seg[dataset_name] = performance
            with open(result_csv, 'a', newline='') as f:
                csv.writer(f).writerow([dataset_name, 'transunet_seg_ddp', performance,
                                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])

    # ===== Classification =====
    for dataset_name in cls_test_set:
        num_classes = 4 if dataset_name == "private_Breast_luminal" else 2

        ds = USdatasetClsFlexible(
            base_dir=os.path.join(args.root_path, "classification", dataset_name),
            split=["test"],
            list_dir=os.path.join(args.root_path, "classification", dataset_name),
            transform=CenterCropGenerator(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt
        )
        sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if dist.is_initialized() else None
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False if sampler else False,
                            num_workers=2, pin_memory=True, sampler=sampler)

        if is_main_process():
            logging.info("%s: %d samples", dataset_name, len(ds))

        model.eval()
        label_list = []
        prediction_list = []
        iterator = tqdm(enumerate(loader), total=len(loader), disable=not is_main_process())

        for i_batch, sampled_batch in iterator:
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name']
            case_name = case_name[0] if isinstance(case_name, (list, tuple)) else str(case_name)
            image = to_chw(image).to(device, non_blocking=True)

            if args.prompt:
                position_prompt = torch.tensor(np.array(sampled_batch['position_prompt'])).permute([1, 0]).float()
                task_prompt = torch.tensor(np.array([[0], [1]])).permute([1, 0]).float()
                type_prompt = torch.tensor(np.array(sampled_batch['type_prompt'])).permute([1, 0]).float()
                nature_prompt = torch.tensor(np.array(sampled_batch['nature_prompt'])).permute([1, 0]).float()
                with torch.no_grad():
                    outputs = model(image, position_prompt.cuda(), task_prompt.cuda(),
                                   type_prompt.cuda(), nature_prompt.cuda())
            else:
                with torch.no_grad():
                    outputs = model(image)

            logits = outputs[2] if num_classes == 4 else outputs[1]
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            label_list.append(label.detach().cpu())
            prediction_list.append(preds.detach().cpu())

        # gather & concat across ranks
        labels = torch.cat(label_list, dim=0) if len(label_list) else torch.empty(0, dtype=torch.long)
        preds  = torch.cat(prediction_list, dim=0) if len(prediction_list) else torch.empty(0, dtype=torch.long)

        if dist.is_initialized():
            # 放到 GPU（NCCL 只接受 CUDA 张量）
            labels = labels.to(device, non_blocking=True)
            preds  = preds.to(device, non_blocking=True)

            gather_labels = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
            gather_preds  = [torch.empty_like(preds)  for _ in range(dist.get_world_size())]
            dist.all_gather(gather_labels, labels)
            dist.all_gather(gather_preds,  preds)

            labels = torch.cat(gather_labels, dim=0)
            preds  = torch.cat(gather_preds,  dim=0)

        # 现在再转回 CPU/Numpy
        labels = labels.detach().cpu().numpy()
        preds  = preds.detach().cpu().numpy()


        if labels.size > 0:
            performance = accuracy_score(labels, preds)
        else:
            performance = 0.0

        if is_main_process():
            logging.info('[CLS] %s acc: %.4f', dataset_name, performance)
            if "private_" in dataset_name:
                private_performence_cls[dataset_name] = performance
            with open(result_csv, 'a', newline='') as f:
                csv.writer(f).writerow([dataset_name, 'transunet_cls_ddp', performance,
                                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])

    # ===== Summary =====
    if is_main_process():
        if private_performence_seg:
            logging.info("Private segmentation performance:")
            for k, v in private_performence_seg.items():
                logging.info("%s: %.4f", k, v)
            logging.info("Mean Dice: %.4f", np.mean(list(private_performence_seg.values())))
        if private_performence_cls:
            logging.info("Private classification performance:")
            for k, v in private_performence_cls.items():
                logging.info("%s: %.4f", k, v)
            logging.info("Mean Acc: %.4f", np.mean(list(private_performence_cls.values())))


def build_transunet_and_load(args, device):
    # 1) 配置
    cfg = VIT_CONFIGS[args.vit_name]
    if hasattr(cfg, "patches") and getattr(cfg.patches, "grid", None) is not None:
        if args.img_size == 224:
            try:
                if isinstance(cfg.patches, dict):
                    gh, gw = cfg.patches["grid"]
                else:
                    gh, gw = cfg.patches.grid
                if (gh, gw) != (14, 14):
                    if isinstance(cfg.patches, dict):
                        cfg.patches["grid"] = (14, 14)
                    else:
                        cfg.patches.grid = (14, 14)
            except Exception:
                pass
    cfg.n_classes = 2

    # 2) 模型
    net = VisionTransformer(
        config=cfg,
        img_size=args.img_size,
        num_classes=cfg.n_classes,
        zero_head=False,
        vis=False,
        prompt=args.prompt,
        pos_len=POSITION_LEN,
        task_len=TASK_LEN,
        type_len=TYPE_LEN,
        nature_len=NAT_LEN
    ).cuda()

    # 3) npz 预训练
    npz_path = args.vit_pretrained or getattr(cfg, "pretrained_path", None)
    if npz_path and os.path.isfile(npz_path):
        if is_main_process():
            print(f"[TransUNet] Loading pretrained (npz) from {npz_path}")
        weights = np.load(npz_path, allow_pickle=True)
        net.load_from(weights)

    # 4) 本地 checkpoint（.pth）
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Resume checkpoint file not found: {args.resume}")
    if is_main_process():
        print("Loading checkpoint from ", args.resume)
    payload = torch.load(args.resume, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict) else payload

    new_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_state[nk] = v
    missing, unexpected = net.load_state_dict(new_state, strict=False)
    if is_main_process():
        print(f"[TransUNet] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    return net


if __name__ == "__main__":
    # ===== 随机数 & CUDNN =====
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ===== DDP =====
    local_rank, device = setup_ddp()

    # ===== 模型 & DDP 包装 =====
    net = build_transunet_and_load(args, device)
    if dist.is_initialized():
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    # ===== 日志（仅 rank0 写文件）=====
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, "test_result.txt")
        with open(log_path, "a"):
            pass
        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(os.path.basename(args.resume))

    # ===== 保存输出目录 =====
    if args.is_saveout:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        if is_main_process():
            os.makedirs(args.test_save_dir, exist_ok=True)
        test_save_path = args.test_save_dir
    else:
        test_save_path = None

    # 同步一下目录创建
    if dist.is_initialized():
        dist.barrier()

    # ===== 推理 =====
    inference(args, net, device, test_save_path)

    # 关闭进程组
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
