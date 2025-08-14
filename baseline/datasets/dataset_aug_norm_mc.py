import os
import cv2
import torch
import numpy as np
import random
from scipy.ndimage import zoom
from scipy import ndimage
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

from datasets.omni_dataset import position_prompt_dict
from datasets.omni_dataset import nature_prompt_dict

from datasets.omni_dataset import position_prompt_one_hot_dict
from datasets.omni_dataset import nature_prompt_one_hot_dict
from datasets.omni_dataset import type_prompt_one_hot_dict

def random_horizontal_flip(image, label):
    axis = 1
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, angle=None):
    angle = np.random.randint(-30, 30) if angle is None else angle
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_gamma_correction(image):
    """Apply random gamma correction to image"""
    gamma = random.uniform(0.8, 1.2)
    img_gamma = np.power(image, gamma)
    img_gamma = np.clip(img_gamma, 0, 1)
    return img_gamma


def random_contrast_adjustment(image):
    """Apply random contrast adjustment to image"""
    alpha = random.uniform(0.8, 1.2)
    img_contrast = image * alpha
    img_contrast = np.clip(img_contrast, 0, 1)
    return img_contrast


def random_vertical_flip(image, label):
    """Apply vertical flip to image and label"""
    axis = 0
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


class RandomGenerator_Seg(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.image_size = output_size

    def __call__(self, sample):
        image, label, nature = sample['image'], sample['label'], sample['nature_for_aug']
        if 'type_prompt' in sample:
            type_prompt = sample['type_prompt']

        # Apply photometric transformations
        # Gamma correction
        if random.random() > 0.5:
            image = random_gamma_correction(image)
        # Contrast adjustment
        if random.random() > 0.5:
            image = random_contrast_adjustment(image)

        # Apply geometric transformations
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        # For tumor type, add conditional flipping
        if nature == "tumor":
            if random.random() > 0.5:
                image, label = random_horizontal_flip(image, label)
            if random.random() > 0.5:
                image, label = random_vertical_flip(image, label)

        # Convert numpy arrays to PIL Images for torchvision transforms
        # Ensure image is in [0, 255] range for PIL
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        label_pil = Image.fromarray(label.astype(np.uint8))

        # Random crop with scaling
        if random.random() > 0.5:
            scale = np.random.uniform(1, 1.2)
            h, w = self.image_size
            new_h, new_w = int(h * scale), int(w * scale)
            image_pil = TF.resize(image_pil, (new_h, new_w), interpolation=Image.BILINEAR)
            label_pil = TF.resize(label_pil, (new_h, new_w), interpolation=Image.NEAREST)
            i, j, crop_h, crop_w = T.RandomCrop.get_params(image_pil, self.image_size)
            image_pil = TF.crop(image_pil, i, j, crop_h, crop_w)
            label_pil = TF.crop(label_pil, i, j, crop_h, crop_w)
        else:
            # Just resize to target size
            image_pil = TF.resize(image_pil, self.image_size, interpolation=Image.BILINEAR)
            label_pil = TF.resize(label_pil, self.image_size, interpolation=Image.NEAREST)

        # Convert back to numpy arrays
        image = np.array(image_pil).astype(np.float32) / 255.0
        label = np.array(label_pil).astype(np.float32)

        # 转换为tensor并进行ImageNet标准化
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # HWC -> CHW
        
        # ImageNet标准化参数
        IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
        image = image.permute(1, 2, 0)  # CHW -> HWC

        # 不需要unsqueeze(0)，DataLoader会自动处理batch维度
        label = torch.from_numpy(label.astype(np.float32))
        if 'type_prompt' in sample:
            sample = {'image': image, 'label': label.long(), 'type_prompt': type_prompt}
        else:
            sample = {'image': image, 'label': label.long()}
        return sample
    

class RandomGenerator_Cls(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.image_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if 'type_prompt' in sample:
            type_prompt = sample['type_prompt']

        # Apply photometric transformations
        # Gamma correction
        if random.random() > 0.5:
            image = random_gamma_correction(image)
        
        # Contrast adjustment
        if random.random() > 0.5:
            image = random_contrast_adjustment(image)

        # Apply geometric transformations
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:
            image, label = random_horizontal_flip(image, label)
        if random.random() > 0.5:
            image, label = random_vertical_flip(image, label)

        # Convert numpy arrays to PIL Images for torchvision transforms
        # Ensure image is in [0, 255] range for PIL
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        label_pil = Image.fromarray(label.astype(np.uint8))

        # Random crop with scaling
        if random.random() > 0.5:
            scale = np.random.uniform(1, 1.2)
            h, w = self.image_size
            new_h, new_w = int(h * scale), int(w * scale)
            image_pil = TF.resize(image_pil, (new_h, new_w), interpolation=Image.BILINEAR)
            label_pil = TF.resize(label_pil, (new_h, new_w), interpolation=Image.NEAREST)
            i, j, crop_h, crop_w = T.RandomCrop.get_params(image_pil, self.image_size)
            image_pil = TF.crop(image_pil, i, j, crop_h, crop_w)
            label_pil = TF.crop(label_pil, i, j, crop_h, crop_w)
        else:
            # Just resize to target size
            image_pil = TF.resize(image_pil, self.image_size, interpolation=Image.BILINEAR)
            label_pil = TF.resize(label_pil, self.image_size, interpolation=Image.NEAREST)

        # Convert back to numpy arrays
        image = np.array(image_pil).astype(np.float32) / 255.0
        label = np.array(label_pil).astype(np.float32)

        # 转换为tensor并进行ImageNet标准化
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # HWC -> CHW
        
        # ImageNet标准化参数
        IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # 应用标准化: (image - mean) / std
        image = (image - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
        # 不需要unsqueeze(0)，DataLoader会自动处理batch维度
        image = image.permute(1, 2, 0) # CHW -> HWC

        label = torch.from_numpy(label.astype(np.float32))
        if 'type_prompt' in sample:
            sample = {'image': image, 'label': label.long(), 'type_prompt': type_prompt}
        else:
            sample = {'image': image, 'label': label.long()}
        return sample


class CenterCropGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if 'type_prompt' in sample:
            type_prompt = sample['type_prompt']
        
        # Convert numpy arrays to PIL Images for torchvision transforms
        # Ensure image is in [0, 255] range for PIL
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        label_pil = Image.fromarray(label.astype(np.uint8))
        
        # For testing, just resize without cropping
        image_pil = TF.resize(image_pil, self.output_size, interpolation=Image.BILINEAR)
        label_pil = TF.resize(label_pil, self.output_size, interpolation=Image.NEAREST)
        
        # Convert back to numpy arrays
        image = np.array(image_pil).astype(np.float32) / 255.0
        label = np.array(label_pil).astype(np.float32)

        # 转换为tensor并进行ImageNet标准化
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # HWC -> CHW
        
        # ImageNet标准化参数
        IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # 应用标准化: (image - mean) / std
        image = (image - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
        image = image.permute(1, 2, 0)  # CHW -> HWC
        
        # 不需要unsqueeze(0)，DataLoader会自动处理batch维度
        label = torch.from_numpy(label.astype(np.float32))
        if 'type_prompt' in sample:
            sample = {'image': image, 'label': label.long(), 'type_prompt': type_prompt}
        else:
            sample = {'image': image, 'label': label.long()}
        return sample
    
class CenterCropGenerator_tta(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if 'type_prompt' in sample:
            type_prompt = sample['type_prompt']
        
        # Convert numpy arrays to PIL Images for torchvision transforms
        # Ensure image is in [0, 255] range for PIL
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        label_pil = Image.fromarray(label.astype(np.uint8))
        
        # For testing, just resize without cropping
        image_pil = TF.resize(image_pil, self.output_size, interpolation=Image.BILINEAR)
        label_pil = TF.resize(label_pil, self.output_size, interpolation=Image.NEAREST)
        
        # Convert back to numpy arrays
        image = np.array(image_pil).astype(np.float32) / 255.0
        label = np.array(label_pil).astype(np.float32)

        # 转换为tensor并进行ImageNet标准化
        image = torch.from_numpy(image.astype(np.float32))
        
        # 不需要unsqueeze(0)，DataLoader会自动处理batch维度
        label = torch.from_numpy(label.astype(np.float32))
        if 'type_prompt' in sample:
            sample = {'image': image, 'label': label.long(), 'type_prompt': type_prompt}
        else:
            sample = {'image': image, 'label': label.long()}
        return sample
    
class USdatasetClsFlexible(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, prompt=False):
        """
        Enhanced USdataset for classification that supports flexible split options.
        
        Args:
            base_dir: Base directory containing the data
            list_dir: Directory containing split files and config.yaml
            split: Can be either:
                   - String: single split name like "val" or "test"
                   - List: multiple split names like ["val", "test"]
            transform: Data transformation function
            prompt: Whether to use prompt features
        
        Usage examples:
            # Single split
            dataset = USdatasetClsFlexible(base_dir, list_dir, "val")
            
            # Multiple splits
            dataset = USdatasetClsFlexible(base_dir, list_dir, ["val", "test"])
        """
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        self.prompt = prompt
        
        # Load config file
        config_path = os.path.join(list_dir, "config.yaml")
        if os.path.exists(config_path):
            self.label_info = open(config_path).readlines()
        else:
            print(f"Warning: Config file {config_path} not found!")
            self.label_info = []
        
        # Handle both single split and multiple splits
        self.sample_list = []
        self.split_info = {}  # Track which samples come from which split
        
        if isinstance(split, str):
            # Single split
            split_file = os.path.join(list_dir, f"{split}.txt")
            if os.path.exists(split_file):
                split_samples = open(split_file).readlines()
                self.sample_list = split_samples
                self.split_info = {i: split for i in range(len(split_samples))}
                print(f"Loaded {len(split_samples)} samples from {split}")
            else:
                print(f"Warning: Split file {split_file} not found!")
                
        elif isinstance(split, list):
            # Multiple splits
            current_idx = 0
            for split_name in split:
                split_file = os.path.join(list_dir, f"{split_name}.txt")
                if os.path.exists(split_file):
                    split_samples = open(split_file).readlines()
                    self.sample_list.extend(split_samples)
                    # Record which split each sample comes from
                    for i in range(len(split_samples)):
                        self.split_info[current_idx + i] = split_name
                    current_idx += len(split_samples)
                    print(f"Loaded {len(split_samples)} samples from {split_name}")
                else:
                    print(f"Warning: Split file {split_file} not found!")
        else:
            raise ValueError("split must be either a string or a list of strings")
        
        print(f"Total samples loaded: {len(self.sample_list)}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_name = self.sample_list[idx].strip('\n')
        img_path = os.path.join(self.data_dir, img_name)

        # Check if image exists
        if not os.path.exists(img_path):
            raise ValueError(f"Image file not found: {img_path}")

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        label = int(img_name.split("/")[0])

        sample = {'image': image/255.0, 'label': np.zeros(image.shape[:2])}
        if self.transform:
            sample = self.transform(sample)
        sample['label'] = torch.from_numpy(np.array(label))
        sample['case_name'] = self.sample_list[idx].strip('\n')
        
        # Add split information to sample
        sample['split_name'] = self.split_info.get(idx, 'unknown')
        sample['nature_for_aug'] = nature_prompt_dict[img_path.split("/")[-3]]
        
        if self.prompt:
            dataset_name = img_path.split("/")[-3]
            sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
            sample['nature_prompt'] = nature_prompt_one_hot_dict[nature_prompt_dict[dataset_name]]
            sample['position_prompt'] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        
        return sample

    def get_split_statistics(self):
        """
        Return statistics about the loaded splits.
        
        Returns:
            dict: Dictionary with split names as keys and sample counts as values
        """
        stats = {}
        for split_name in self.split_info.values():
            stats[split_name] = stats.get(split_name, 0) + 1
        return stats

    def get_samples_by_split(self, split_name):
        """
        Get indices of samples that belong to a specific split.
        
        Args:
            split_name (str): Name of the split
            
        Returns:
            list: List of indices belonging to the specified split
        """
        indices = []
        for idx, sname in self.split_info.items():
            if sname == split_name:
                indices.append(idx)
        return indices

class USdatasetSegFlexible(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, prompt=False):
        """
        Enhanced USdataset for segmentation that supports flexible split options.
        
        Args:
            base_dir: Base directory containing the data
            list_dir: Directory containing split files and config.yaml
            split: Can be either:
                   - String: single split name like "val" or "test"
                   - List: multiple split names like ["val", "test"]
            transform: Data transformation function
            prompt: Whether to use prompt features
        
        Usage examples:
            # Single split
            dataset = USdatasetSegFlexible(base_dir, list_dir, "val")
            
            # Multiple splits
            dataset = USdatasetSegFlexible(base_dir, list_dir, ["val", "test"])
        """
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        self.prompt = prompt
        
        # Load config file
        config_path = os.path.join(list_dir, "config.yaml")
        if os.path.exists(config_path):
            self.label_info = open(config_path).readlines()
        else:
            print(f"Warning: Config file {config_path} not found!")
            self.label_info = []
        
        # Handle both single split and multiple splits
        self.sample_list = []
        self.split_info = {}  # Track which samples come from which split
        
        if isinstance(split, str):
            # Single split
            split_file = os.path.join(list_dir, f"{split}.txt")
            if os.path.exists(split_file):
                split_samples = open(split_file).readlines()
                self.sample_list = split_samples
                self.split_info = {i: split for i in range(len(split_samples))}
                print(f"Loaded {len(split_samples)} samples from {split}")
            else:
                print(f"Warning: Split file {split_file} not found!")
                
        elif isinstance(split, list):
            # Multiple splits
            current_idx = 0
            for split_name in split:
                split_file = os.path.join(list_dir, f"{split_name}.txt")
                if os.path.exists(split_file):
                    split_samples = open(split_file).readlines()
                    self.sample_list.extend(split_samples)
                    # Record which split each sample comes from
                    for i in range(len(split_samples)):
                        self.split_info[current_idx + i] = split_name
                    current_idx += len(split_samples)
                    print(f"Loaded {len(split_samples)} samples from {split_name}")
                else:
                    print(f"Warning: Split file {split_file} not found!")
        else:
            raise ValueError("split must be either a string or a list of strings")
        
        print(f"Total samples loaded: {len(self.sample_list)}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_name = self.sample_list[idx].strip('\n')
        img_path = os.path.join(self.data_dir, "imgs", img_name)
        label_path = os.path.join(self.data_dir, "masks", img_name)

        # Check if files exist
        if not os.path.exists(img_path):
            raise ValueError(f"Image file not found: {img_path}")
        if not os.path.exists(label_path):
            raise ValueError(f"Label file not found: {label_path}")

        image = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        if label is None:
            raise ValueError(f"Could not load label: {label_path}")

        # Process label according to config
        label_info_list = [info.strip().split(":") for info in self.label_info]
        for single_label_info in label_info_list:
            if len(single_label_info) >= 3:
                label_index = int(single_label_info[0])
                label_value_in_image = int(single_label_info[2])
                label[label == label_value_in_image] = label_index

        label[label > 0] = 1

        sample = {'image': image/255.0, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        
        # Add split information to sample
        sample['split_name'] = self.split_info.get(idx, 'unknown')
        sample['nature_for_aug'] = nature_prompt_dict[img_path.split("/")[-3]]
        if self.prompt:
            dataset_name = img_path.split("/")[-3]
            sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
            sample['nature_prompt'] = nature_prompt_one_hot_dict[nature_prompt_dict[dataset_name]]
            sample['position_prompt'] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        
        return sample

    def get_split_statistics(self):
        """
        Return statistics about the loaded splits.
        
        Returns:
            dict: Dictionary with split names as keys and sample counts as values
        """
        stats = {}
        for split_name in self.split_info.values():
            stats[split_name] = stats.get(split_name, 0) + 1
        return stats

    def get_samples_by_split(self, split_name):
        """
        Get indices of samples that belong to a specific split.
        
        Args:
            split_name (str): Name of the split
            
        Returns:
            list: List of indices belonging to the specified split
        """
        indices = []
        for idx, sname in self.split_info.items():
            if sname == split_name:
                indices.append(idx)
        return indices