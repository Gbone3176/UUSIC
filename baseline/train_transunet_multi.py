import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Datasets & augs
from datasets.dataset_aug_norm import (
    USdatasetCls,
    USdatasetSeg,
    RandomGenerator_Cls,
    RandomGenerator_Seg,
    CenterCropGenerator,
)
from datasets.omni_dataset import USdatasetOmni_cls, USdatasetOmni_seg

# Model (TransUNet)
from networks.transUnet.vit_seg_modeling import VisionTransformer, Transformer, CONFIGS

# Utils
from utils import DiceLoss


def build_seg_model(cfg_name: str, img_size: int, n_classes: int, pretrained_npz: str | None):
    config = CONFIGS[cfg_name]
    config.n_classes = n_classes
    model = VisionTransformer(config, img_size=img_size, num_classes=n_classes, zero_head=False, vis=False)

    # Load ViT weights from .npz if provided or from config
    npz_path = pretrained_npz or getattr(config, 'pretrained_path', None)
    if npz_path and os.path.exists(npz_path):
        import numpy as np
        weights = np.load(npz_path)
        model.load_from(weights)
    return model


class TransUNetClassifier(nn.Module):
    """Classifier built on top of the same ViT backbone used by TransUNet.

    It uses the Transformer encoder to extract patch embeddings and global-average pools
    them to get an image representation, then applies two heads:
    - 2-way classification head (for binary datasets)
    - 4-way classification head (for 4-class datasets like private_Breast_luminal)
    """
    def __init__(self, cfg_name: str, img_size: int, pretrained_npz: str | None):
        super().__init__()
        config = CONFIGS[cfg_name]
        self.hidden_size = config.hidden_size
        self.transformer = Transformer(config, img_size=img_size, vis=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Linear(self.hidden_size, 2)
        self.fc4 = nn.Linear(self.hidden_size, 4)

        # Initialize from npz like TransUNet
        npz_path = pretrained_npz or getattr(config, 'pretrained_path', None)
        if npz_path and os.path.exists(npz_path):
            import numpy as np
            weights = np.load(npz_path)
            # Reuse VisionTransformer loader to fill transformer weights
            vt = VisionTransformer(config, img_size=img_size, num_classes=2, zero_head=False, vis=False)
            vt.load_from(weights)
            # copy encoder+embedding weights
            self.transformer.load_state_dict(vt.transformer.state_dict())

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        encoded, _, _ = self.transformer(x)  # (B, n_patches, hidden)
        feats = encoded.mean(dim=1)          # GAP over patches
        logits_2 = self.fc2(feats)
        logits_4 = self.fc4(feats)
        return logits_2, logits_4


def get_dataloaders(args):
    # Segmentation: use omni mixture of private sets by default
    if args.seg_dataset == 'omni':
        ds_train_seg = USdatasetOmni_seg(base_dir=args.root_path, split='train',
                                         transform=RandomGenerator_Seg([args.img_size, args.img_size]),
                                         prompt=False)
        ds_val_seg_list = [
            ('private_Thyroid', 2),
            ('private_Kidney', 2),
            ('private_Fetal_Head', 2),
            ('private_Cardiac', 2),
            ('private_Breast_luminal', 2),
            ('private_Breast', 2),
        ]
        val_seg_loaders = []
        for name, num_classes in ds_val_seg_list:
            val_ds = USdatasetSeg(
                base_dir=os.path.join(args.root_path, 'segmentation', name),
                list_dir=os.path.join(args.root_path, 'segmentation', name),
                split='test',
                transform=CenterCropGenerator([args.img_size, args.img_size]),
                prompt=False,
            )
            val_seg_loaders.append((name, num_classes, DataLoader(val_ds, batch_size=args.batch_size,
                                                                  shuffle=False, num_workers=args.num_workers,
                                                                  pin_memory=True)))
    else:
        # Single dataset
        base = os.path.join(args.root_path, 'segmentation', args.seg_dataset)
        ds_train_seg = USdatasetSeg(base_dir=base, list_dir=base, split='train',
                                    transform=RandomGenerator_Seg([args.img_size, args.img_size]),
                                    prompt=False)
        val_ds = USdatasetSeg(base_dir=base, list_dir=base, split='val',
                               transform=CenterCropGenerator([args.img_size, args.img_size]), prompt=False)
        val_seg_loaders = [(args.seg_dataset, args.seg_num_classes,
                            DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=True))]

    trainloader_seg = DataLoader(ds_train_seg, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=True)

    # Classification: omni by default
    if args.cls_dataset == 'omni':
        ds_train_cls = USdatasetOmni_cls(base_dir=args.root_path, split='train',
                                         transform=RandomGenerator_Cls([args.img_size, args.img_size]),
                                         prompt=False)
        # Validation sets
        cls_val_sets = ['private_Liver', 'private_Breast_luminal', 'private_Breast', 'private_Appendix']
        val_cls_loaders = []
        for name in cls_val_sets:
            ds = USdatasetCls(
                base_dir=os.path.join(args.root_path, 'classification', name),
                list_dir=os.path.join(args.root_path, 'classification', name),
                split='test',
                transform=CenterCropGenerator([args.img_size, args.img_size]),
                prompt=False,
            )
            val_cls_loaders.append((name, DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.num_workers, pin_memory=True)))
    else:
        base = os.path.join(args.root_path, 'classification', args.cls_dataset)
        ds_train_cls = USdatasetCls(base_dir=base, list_dir=base, split='train',
                                    transform=RandomGenerator_Cls([args.img_size, args.img_size]), prompt=False)
        val_ds = USdatasetCls(base_dir=base, list_dir=base, split='val',
                              transform=CenterCropGenerator([args.img_size, args.img_size]), prompt=False)
        val_cls_loaders = [(args.cls_dataset, DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                                         num_workers=args.num_workers, pin_memory=True))]

    trainloader_cls = DataLoader(ds_train_cls, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=True)

    return trainloader_seg, trainloader_cls, val_seg_loaders, val_cls_loaders


def evaluate_seg(seg_model, val_seg_loaders, device):
    from utils import omni_seg_test  # existing helper for Dice
    seg_model.eval()
    results = []
    with torch.no_grad():
        for name, num_classes, loader in val_seg_loaders:
            total = 0.0
            count = 0
            for batch in loader:
                img, lab = batch['image'].to(device), batch['label'].to(device)
                # omni_seg_test expects model with same interface as training
                metric_i = omni_seg_test(img, lab, seg_model, classes=num_classes)
                # adapt returned format (list of (values, valid_flags)) -> average single class
                metric_i = metric_i[0][0]  # list of per-sample dice
                total += float(torch.tensor(metric_i).sum())
                count += len(metric_i)
            dice = total / max(count, 1)
            results.append((name, dice))
    seg_model.train()
    return results


def evaluate_cls(cls_model, val_cls_loaders, device):
    from sklearn.metrics import roc_auc_score
    import numpy as np

    cls_model.eval()
    results = []
    with torch.no_grad():
        for name, loader in val_cls_loaders:
            probs_all = []
            labels_all = []
            num_classes = 4 if 'Breast_luminal' in name or 'Breast_luminal' in name else 2
            for batch in loader:
                img, label = batch['image'].to(device), batch['label']
                logits_2, logits_4 = cls_model(img)
                logits = logits_4 if num_classes == 4 else logits_2
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_all.append(probs)
                labels_all.append(label.numpy())
            import numpy as np
            probs_all = np.concatenate(probs_all, axis=0)
            labels_all = np.concatenate(labels_all, axis=0)
            labels_oh = np.eye(num_classes)[labels_all]
            try:
                auc = roc_auc_score(labels_oh, probs_all, multi_class='ovo' if num_classes > 2 else 'raise')
            except Exception:
                # Fallback to accuracy if AUC not defined for binary single-class fold
                auc = (probs_all.argmax(axis=1) == labels_all).mean()
            results.append((name, float(auc)))
    cls_model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description='Train TransUNet for segmentation + classification (multi-task).')
    parser.add_argument('--root_path', type=str, default='data/', help='Data root')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--cfg', type=str, default='R50-ViT-B_16', choices=list(CONFIGS.keys()))
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pretrained_npz', type=str, default=None, help='Path to ViT .npz weights')

    # dataset choices (default: mix all private datasets)
    parser.add_argument('--seg_dataset', type=str, default='omni', help='omni or a specific dataset name under data/segmentation')
    parser.add_argument('--seg_num_classes', type=int, default=2)
    parser.add_argument('--cls_dataset', type=str, default='omni', help='omni or a specific dataset name under data/classification')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Models
    seg_model = build_seg_model(args.cfg, args.img_size, args.seg_num_classes, args.pretrained_npz).to(device)
    cls_model = TransUNetClassifier(args.cfg, args.img_size, args.pretrained_npz).to(device)

    # Losses
    seg_ce = nn.CrossEntropyLoss()
    seg_dice = DiceLoss()
    cls_ce_2 = nn.CrossEntropyLoss()
    cls_ce_4 = nn.CrossEntropyLoss()

    # Optimizer
    params = list(seg_model.parameters()) + list(cls_model.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Data
    trainloader_seg, trainloader_cls, val_seg_loaders, val_cls_loaders = get_dataloaders(args)

    # Logging
    writer = SummaryWriter(os.path.join(args.output_dir, 'tb'))

    global_step = 0
    best_seg = -1.0
    best_cls = -1.0

    for epoch in range(1, args.epochs + 1):
        seg_model.train(); cls_model.train()
        # alternate training between seg and cls
        seg_it = iter(trainloader_seg)
        cls_it = iter(trainloader_cls)
        steps = max(len(trainloader_seg), len(trainloader_cls))
        for step in range(steps):
            # 1) segmentation step (if available)
            if step < len(trainloader_seg):
                try:
                    batch = next(seg_it)
                except StopIteration:
                    seg_it = iter(trainloader_seg)
                    batch = next(seg_it)
                img, lab = batch['image'].to(device), batch['label'].to(device)
                logits = seg_model(img)
                loss_seg = 0.28 * seg_ce(logits, lab.long()) + 0.72 * seg_dice(logits, lab, softmax=True)
                optimizer.zero_grad()
                loss_seg.backward()
                optimizer.step()
                writer.add_scalar('train/seg_loss', loss_seg.item(), global_step)
                global_step += 1

            # 2) classification step (if available)
            if step < len(trainloader_cls):
                try:
                    batch = next(cls_it)
                except StopIteration:
                    cls_it = iter(trainloader_cls)
                    batch = next(cls_it)
                img, labels = batch['image'].to(device), batch['label'].to(device)
                num_classes_batch = batch.get('num_classes', None)
                logits_2, logits_4 = cls_model(img)
                loss_cls = torch.tensor(0.0, device=device)
                if num_classes_batch is None:
                    # assume 2-way only if dataset does not provide metadata
                    loss_cls = cls_ce_2(logits_2, labels.long())
                else:
                    mask_2 = (num_classes_batch == 2)
                    mask_4 = (num_classes_batch == 4)
                    if mask_2.any():
                        loss_cls = loss_cls + cls_ce_2(logits_2[mask_2], labels[mask_2].long())
                    if mask_4.any():
                        loss_cls = loss_cls + cls_ce_4(logits_4[mask_4], labels[mask_4].long())
                optimizer.zero_grad()
                loss_cls.backward()
                optimizer.step()
                writer.add_scalar('train/cls_loss', loss_cls.item(), global_step)
                global_step += 1

        # ---- validation ----
        if epoch % 1 == 0:
            seg_res = evaluate_seg(seg_model, val_seg_loaders, device)
            cls_res = evaluate_cls(cls_model, val_cls_loaders, device)
            # log
            for name, dice in seg_res:
                writer.add_scalar(f'val_seg/{name}_dice', dice, epoch)
            for name, auc in cls_res:
                writer.add_scalar(f'val_cls/{name}_auc', auc, epoch)

            # checkpointing (simple best-on-avg)
            avg_dice = sum(d for _, d in seg_res) / max(len(seg_res), 1)
            avg_auc = sum(a for _, a in cls_res) / max(len(cls_res), 1)
            if avg_dice > best_seg or avg_auc > best_cls:
                best_seg = max(best_seg, avg_dice)
                best_cls = max(best_cls, avg_auc)
                ckpt = {
                    'epoch': epoch,
                    'seg_model': seg_model.state_dict(),
                    'cls_model': cls_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_seg': best_seg,
                    'best_cls': best_cls,
                }
                path = os.path.join(args.output_dir, f'best_epoch{epoch}_dice{avg_dice:.4f}_auc{avg_auc:.4f}.pth')
                torch.save(ckpt, path)

        # save latest
        if epoch % 5 == 0:
            ckpt = {
                'epoch': epoch,
                'seg_model': seg_model.state_dict(),
                'cls_model': cls_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'latest.pth'))

    writer.close()
    print('Training done.')


if __name__ == '__main__':
    main()
