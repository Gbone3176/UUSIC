import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from omni_trainer_TU_v2_mix import omni_train
from networks.vit_seg_modeling_v2 import VisionTransformer, CONFIGS as VIT_CONFIGS
from datasets.omni_dataset import position_prompt_one_hot_dict, task_prompt_one_hot_dict, type_prompt_one_hot_dict, nature_prompt_one_hot_dict

POSITION_LEN = len(position_prompt_one_hot_dict)
TASK_LEN = len(task_prompt_one_hot_dict)
TYPE_LEN = len(type_prompt_one_hot_dict)
NAT_LEN = len(nature_prompt_one_hot_dict)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='data/', help='root dir for data')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--resume', help='resume from checkpoint')

# 你自己的 .pth checkpoint（断点续训）
parser.add_argument('--pretrain_ckpt', type=str, help='pretrained checkpoint (.pth)')

# ====== ViT 新增参数 ======
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                    choices=['ViT-B_16','ViT-B_32','ViT-L_16','ViT-L_32','ViT-H_14','R50-ViT-B_16','R50-ViT-L_16'],
                    help='ViT backbone name')
parser.add_argument('--vit_pretrained', type=str, default="/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/pretrained_ckpt/R50+ViT-B_16.npz",
                    help='.npz pretrained path for ViT (optional)')
parser.add_argument('--n_classes_seg', type=int, default=2, help='num classes for segmentation head')

# prompt/adapter 关闭
parser.add_argument('--prompt', action='store_true', help='(deprecated here) using prompt for training')
parser.add_argument('--adapter_ft', action='store_true', help='(deprecated here) using adapter for fine-tuning')

# ====== Mixup and CutMix 参数 ======
parser.add_argument('--mixup_alpha', type=float, default=0.8, help='mixup alpha parameter')
parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='cutmix alpha parameter')
parser.add_argument('--mixup_prob', type=float, default=1.0, help='probability of applying mixup/cutmix')
parser.add_argument('--cutmix_prob', type=float, default=1.0, help='probability of applying cutmix when both enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='probability of switching to cutmix when both enabled')


args = parser.parse_args()

# 如果你还需要其它 cfg（比如日志用），可以保留
# config = get_config(args)

if __name__ == "__main__":
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

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # ===== Build VisionTransformer for Seg + Cls =====
    vit_cfg = VIT_CONFIGS[args.vit_name]
    vit_cfg.n_classes = args.n_classes_seg

    net = VisionTransformer(
        config=vit_cfg,
        img_size=args.img_size,
        num_classes=vit_cfg.n_classes,
        zero_head=False,
        vis=False,
        prompt=args.prompt,
        pos_len=POSITION_LEN,
        task_len=TASK_LEN,
        type_len=TYPE_LEN,
        nature_len=NAT_LEN
    )

    if args.pretrain_ckpt is not None:
        print(f"Loading pretrained checkpoint from {args.pretrain_ckpt}")
        net.load_state_dict(torch.load(args.pretrain_ckpt, map_location='cpu'), strict=False)

    # ===== 打印 Mixup/CutMix 配置 =====
    print(f"** Mixup/CutMix Configuration **")
    print(f"  mixup_alpha: {args.mixup_alpha}")
    print(f"  cutmix_alpha: {args.cutmix_alpha}")
    print(f"  mixup_prob: {args.mixup_prob}")
    print(f"  cutmix_prob: {args.cutmix_prob}")
    print(f"  mixup_switch_prob: {args.mixup_switch_prob}")

    # ===== 开始训练 =====
    snapshot_path = args.output_dir
    omni_train(args, net, snapshot_path) 