import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from omni_trainer_TU_v4 import omni_train
from networks.vit_seg_modeling_v4 import VisionTransformer, CONFIGS as VIT_CONFIGS
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

# 下面这几个是原来 Swin 的参数（保留无妨，但不会用于建模）
parser.add_argument('--cfg', type=str, default="configs/swin_tiny_patch4_window7_224_lite-PG.yaml",
                    metavar="FILE", help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
                    default=None, nargs='+')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'])

parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

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
    ).cuda()

    # ViT 官方/社区预训练（.npz）
    if args.vit_pretrained is not None and os.path.isfile(args.vit_pretrained):
        print(f"[ViT] Loading pretrained from {args.vit_pretrained}")
        vit_weights = np.load(args.vit_pretrained)
        net.load_from(vit_weights)

    # 你自己的断点续训（.pth）
    if args.pretrain_ckpt is not None and os.path.isfile(args.pretrain_ckpt):
        print(f"[CKPT] Resuming weights from {args.pretrain_ckpt}")
        state = torch.load(args.pretrain_ckpt, map_location='cpu')
        if 'model' in state:
            net.load_state_dict(state['model'], strict=False)
        else:
            for key in list(state.keys()):
                if key.startswith('module.'):
                    state[key[7:]] = state.pop(key)
            ret = net.load_state_dict(state, strict=False)
            print("missing_keys: ", len(ret.missing_keys), "unexpected_keys:", len(ret.unexpected_keys))
    
    # #加载除了分类器之外的其他权重
    # if args.pretrain_ckpt is not None and os.path.isfile(args.pretrain_ckpt):
    #     print(f"[CKPT] Resuming weights (except classifier) from {args.pretrain_ckpt}")
    #     state = torch.load(args.pretrain_ckpt, map_location="cpu")

    #     # 1) 取出 state_dict（兼容 {'model': xxx} 或直接是 dict）
    #     src = state.get("model", state)

    #     # 2) 过滤：去掉 DDP 前缀；丢弃分类器参数；丢弃形状不匹配的键
    #     dst_sd = net.state_dict()
    #     filtered, skipped_cls, skipped_mismatch = {}, [], []

    #     for k, v in src.items():
    #         name = k[7:] if k.startswith("module.") else k

    #         # 跳过分类器
    #         if name.startswith("classifier_head.") or ".classifier_head." in name:
    #             skipped_cls.append(name)
    #             continue

    #         # 只加载存在且形状一致的权重
    #         if name in dst_sd and dst_sd[name].shape == v.shape:
    #             filtered[name] = v
    #         else:
    #             skipped_mismatch.append(name)

    #     # 3) 加载（严格度放宽，允许未匹配到的键）
    #     ret = net.load_state_dict(filtered, strict=False)

    #     # 4) 打印统计
    #     print(f"[CKPT] loaded={len(filtered)} | skipped_cls={len(skipped_cls)} | "
    #         f"skipped_shape_mismatch={len(skipped_mismatch)}")
    #     if ret.missing_keys or ret.unexpected_keys:
    #         print(f"[CKPT] missing_keys={len(ret.missing_keys)} unexpected_keys={len(ret.unexpected_keys)}")

    # 强制关闭 adapter
    args.adapter_ft = False

    omni_train(args, net, args.output_dir)
