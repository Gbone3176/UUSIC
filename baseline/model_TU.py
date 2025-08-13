# sample_result_submission/model.py

import os
import cv2
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import torch

from config_PG import get_config
from networks.vit_seg_modeling_v2 import VisionTransformer, CONFIGS as VIT_CONFIGS
from datasets.dataset_aug_norm import CenterCropGenerator

# -------- Prompt 相关映射 --------
organ_to_position_map = {
    'Breast': 'breast', 'Heart': 'cardiac', 'Thyroid': 'thyroid', 'Head': 'head',
    'Kidney': 'kidney', 'Appendix': 'appendix', 'Liver': 'liver'
}
position_prompt_one_hot_dict = {
    "breast":[1,0,0,0,0,0,0,0], "cardiac":[0,1,0,0,0,0,0,0], "thyroid":[0,0,1,0,0,0,0,0],
    "head":[0,0,0,1,0,0,0,0], "kidney":[0,0,0,0,1,0,0,0], "appendix":[0,0,0,0,0,1,0,0],
    "liver":[0,0,0,0,0,0,1,0], "indis":[0,0,0,0,0,0,0,1]
}
task_prompt_one_hot_dict = {"segmentation":[1,0],"classification":[0,1]}
organ_to_nature_map = {
    'Breast':'tumor','Heart':'organ','Thyroid':'tumor','Head':'organ',
    'Kidney':'organ','Appendix':'organ','Liver':'organ'
}
nature_prompt_one_hot_dict = {"tumor":[1,0],"organ":[0,1]}
type_prompt_one_hot_dict = {"whole":[1,0,0],"local":[0,1,0],"location":[0,0,1]}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    p.add_argument('--data_list', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--checkpoint', required=True, help='model ckpt (.pth)')
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--use_prompts', action='store_true', default=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--vit_type', type=str, default='R50-ViT-B_16',
                   choices=VIT_CONFIGS.keys(),
                   help='Vision Transformer type, e.g., R50-ViT-B_16')
    return p.parse_args()

def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model', ckpt)

    # for i in range(10):
    #     print(state.keys()[i] if i < len(state) else 'No more keys')

    # 去掉或添加 module. 以匹配
    def try_load(sd):
        missing, unexpected = model.load_state_dict(sd, strict=False)
        return missing, unexpected
    
    def strip_module(sd):
        return { (k[7:] if k.startswith('module.') else k): v for k,v in sd.items() }

    try:
        m,u = try_load(strip_module(state))
        print(f'Checkpoint loaded (missing {len(m)}, unexpected {len(u)})')
    except Exception as e:
        print(f'Load attempt failed: {e}')
        
    return ckpt.get('epoch', 0)

def to_chw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous() if x.dim() == 4 and x.size(-1) in (1, 3) else x

class InferenceModel:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        class CfgArgs:
            img_size = args.img_size
            prompt = args.use_prompts
            # 其余训练无关占位
            opts=None; batch_size=None; zip=False; cache_mode=None; resume=None
            accumulation_steps=None; use_checkpoint=False; amp_opt_level=''
            tag=None; eval=True; throughput=False

        cfg_args = CfgArgs()
        config = VIT_CONFIGS[args.vit_type]
        self.net = VisionTransformer(config, prompt=cfg_args.prompt).to(self.device)
        load_checkpoint(self.net, args.checkpoint, self.device)
        self.net.eval()
        self.transform = CenterCropGenerator(output_size=[args.img_size, args.img_size])

    def _build_prompts(self, task, organ):
        # task
        task_vec = task_prompt_one_hot_dict[task]
        task_prompt = torch.tensor(task_vec, dtype=torch.float).unsqueeze(0).to(self.device)
        # position
        pos_key = organ_to_position_map.get(organ, 'indis')
        pos_vec = position_prompt_one_hot_dict[pos_key]
        position_prompt = torch.tensor(pos_vec, dtype=torch.float).unsqueeze(0).to(self.device)
        # nature
        nat_key = organ_to_nature_map.get(organ, 'organ')
        nat_vec = nature_prompt_one_hot_dict[nat_key]
        nature_prompt = torch.tensor(nat_vec, dtype=torch.float).unsqueeze(0).to(self.device)
        # type (固定 whole，可按需逻辑扩展)
        type_vec = type_prompt_one_hot_dict['whole']
        type_prompt = torch.tensor(type_vec, dtype=torch.float).unsqueeze(0).to(self.device)
        return position_prompt, task_prompt, type_prompt, nature_prompt

    def _infer_forward(self, image_tensor, task, organ):

        if self.args.use_prompts:
            pos_p, task_p, type_p, nat_p = self._build_prompts(task, organ)
            # 兼容新 forward：假设 forward(image, position_prompt, task_prompt, type_prompt, nature_prompt,
            #                                   dataset_name=..., task_type=..., use_dataset_specific=True)
            outputs = self.net(
                image_tensor,
                pos_p, task_p, type_p, nat_p
            )
        else:
            outputs = self.net(image_tensor)
        return outputs

    def predict(self, data_list, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        cls_results = {}
        for item in tqdm(data_list, desc='Infer'):
            img_rel = item['img_path_relative']
            img_path = os.path.join(input_dir, img_rel)
            task = item['task']              # 'segmentation' or 'classification'
            dataset_name = item['dataset_name']
            organ = item['organ']

            img = Image.open(img_path).convert('RGB')
            orig_size = img.size  # (W,H)
            img_np = np.array(img)
            sample = {'image': img_np/255.0, 'label': np.zeros(img_np.shape[:2])}
            proc = self.transform(sample)
            # transform 返回 (C,H,W) tensor
            image_tensor = proc['image'].unsqueeze(0).to(self.device)  # [1,C,H,W]
            image_tensor = to_chw(image_tensor)  # 确保是 [1,C,H,W] 格式

            with torch.no_grad():
                outputs = self._infer_forward(image_tensor, task, organ)

            if task == 'classification':
                # 假设 outputs 为字典或 tuple；这里假设返回 dict：{'cls_logits':..., 'seg_logits':...} 或 tuple
                if isinstance(outputs, dict):
                    logits = outputs.get('cls_logits')
                else:
                    # 兼容 tuple: (seg_logits, cls2_logits, cls4_logits, ...)
                    if dataset_name.lower().find('luminal') != -1:
                        logits = outputs[2]  # 4 类
                        num_classes = 4
                    else:
                        logits = outputs[1]  # 2 类
                        num_classes = 2
                probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze(0)
                pred = int(probs.argmax())
                cls_results[img_rel] = {'probability': probs.tolist(), 'prediction': pred, 'num_classes': num_classes}
            else:  # segmentation
                if isinstance(outputs, dict):
                    seg_logits = outputs.get('seg_logits')
                else:
                    seg_logits = outputs[0]
                seg_pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1).squeeze(0)
                mask_small = seg_pred.cpu().numpy().astype(np.uint8)
                # 二值（若多类可直接保存索引，这里按二类转 0/255）
                if mask_small.max() <= 1:
                    mask_small = (mask_small * 255).astype(np.uint8)
                resized = cv2.resize(mask_small, orig_size, interpolation=cv2.INTER_NEAREST)
                out_path = os.path.join(output_dir, img_rel.replace('img', 'mask'))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                Image.fromarray(resized).save(out_path)

        with open(os.path.join(output_dir, 'classification.json'), 'w') as f:
            json.dump(cls_results, f, indent=2)
        print('Done.')

def main():
    args = parse_args()
    with open(args.data_list, 'r') as f:
        data_list = json.load(f)
    model = InferenceModel(args)
    model.predict(data_list, args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()