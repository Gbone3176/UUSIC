import os
import cv2
import random
import numpy as np
import torch
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

        # Resize image and label
        x, y, _ = image.shape

        if x > y:
            image = zoom(image, (self.output_size[0] / y, self.output_size[1] / y, 1), order=1)
            label = zoom(label, (self.output_size[0] / y, self.output_size[1] / y), order=0)
        else:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / x, 1), order=1)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / x), order=0)

        scale = random.uniform(0.8, 1.2)
        image = zoom(image, (scale, scale, 1), order=1)
        label = zoom(label, (scale, scale), order=0)

        x, y, _ = image.shape
        if scale > 1:
            startx = x//2 - (self.output_size[0]//2)
            starty = y//2 - (self.output_size[1]//2)
            image = image[startx:startx+self.output_size[0], starty:starty+self.output_size[1], :]
            label = label[startx:startx+self.output_size[0], starty:starty+self.output_size[1]]
        else:
            if x > self.output_size[0]:
                startx = x//2 - (self.output_size[0]//2)
                image = image[startx:startx+self.output_size[0], :, :]
                label = label[startx:startx+self.output_size[0], :]
            if y > self.output_size[1]:
                starty = y//2 - (self.output_size[1]//2)
                image = image[:, starty:starty+self.output_size[1], :]
                label = label[:, starty:starty+self.output_size[1]]
            x, y, _ = image.shape
            new_image = np.zeros((self.output_size[0], self.output_size[1], 3))
            new_label = np.zeros((self.output_size[0], self.output_size[1]))
            if x < y:
                startx = self.output_size[0]//2 - (x//2)
                starty = 0
                new_image[startx:startx+x, starty:starty+y, :] = image
                new_label[startx:startx+x, starty:starty+y] = label
            else:
                startx = 0
                starty = self.output_size[1]//2 - (y//2)
                new_image[startx:startx+x, starty:starty+y, :] = image
                new_label[startx:startx+x, starty:starty+y] = label
            image = new_image
            label = new_label

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
            image, label = random_rotate(image, label, angle=30)
        if random.random() > 0.5:
            image, label = random_horizontal_flip(image, label)
        if random.random() > 0.5:
            image, label = random_vertical_flip(image, label)

        # Resize image and label
        x, y, _ = image.shape

        if x > y:
            image = zoom(image, (self.output_size[0] / y, self.output_size[1] / y, 1), order=1)
            label = zoom(label, (self.output_size[0] / y, self.output_size[1] / y), order=0)
        else:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / x, 1), order=1)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / x), order=0)

        scale = random.uniform(0.8, 1.2)
        image = zoom(image, (scale, scale, 1), order=1)
        label = zoom(label, (scale, scale), order=0)

        x, y, _ = image.shape
        if scale > 1:
            startx = x//2 - (self.output_size[0]//2)
            starty = y//2 - (self.output_size[1]//2)
            image = image[startx:startx+self.output_size[0], starty:starty+self.output_size[1], :]
            label = label[startx:startx+self.output_size[0], starty:starty+self.output_size[1]]
        else:
            if x > self.output_size[0]:
                startx = x//2 - (self.output_size[0]//2)
                image = image[startx:startx+self.output_size[0], :, :]
                label = label[startx:startx+self.output_size[0], :]
            if y > self.output_size[1]:
                starty = y//2 - (self.output_size[1]//2)
                image = image[:, starty:starty+self.output_size[1], :]
                label = label[:, starty:starty+self.output_size[1]]
            x, y, _ = image.shape
            new_image = np.zeros((self.output_size[0], self.output_size[1], 3))
            new_label = np.zeros((self.output_size[0], self.output_size[1]))
            if x < y:
                startx = self.output_size[0]//2 - (x//2)
                starty = 0
                new_image[startx:startx+x, starty:starty+y, :] = image
                new_label[startx:startx+x, starty:starty+y] = label
            else:
                startx = 0
                starty = self.output_size[1]//2 - (y//2)
                new_image[startx:startx+x, starty:starty+y, :] = image
                new_label[startx:startx+x, starty:starty+y] = label
            image = new_image
            label = new_label

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
        x, y, _ = image.shape
        if x > y:
            image = zoom(image, (self.output_size[0] / y, self.output_size[1] / y, 1), order=1)
            label = zoom(label, (self.output_size[0] / y, self.output_size[1] / y), order=0)
        else:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / x, 1), order=1)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / x), order=0)
        x, y, _ = image.shape
        startx = x//2 - (self.output_size[0]//2)
        starty = y//2 - (self.output_size[1]//2)
        image = image[startx:startx+self.output_size[0], starty:starty+self.output_size[1], :]
        label = label[startx:startx+self.output_size[0], starty:starty+self.output_size[1]]

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


class USdatasetSeg(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, prompt=False):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()

        self.data_dir = base_dir
        self.label_info = open(os.path.join(list_dir, "config.yaml")).readlines()
        self.prompt = prompt

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_name = self.sample_list[idx].strip('\n')
        img_path = os.path.join(self.data_dir, "imgs", img_name)
        label_path = os.path.join(self.data_dir, "masks", img_name)

        image = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        label_info_list = [info.strip().split(":") for info in self.label_info]
        for single_label_info in label_info_list:
            label_index = int(single_label_info[0])
            label_value_in_image = int(single_label_info[2])
            label[label == label_value_in_image] = label_index

        label[label > 0] = 1

        sample = {'image': image/255.0, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        if self.prompt:
            dataset_name = img_path.split("/")[-3]
            sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
            sample['nature_prompt'] = nature_prompt_one_hot_dict[nature_prompt_dict[dataset_name]]
            sample['position_prompt'] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        return sample


class USdatasetCls(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, prompt=False):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()

        self.data_dir = base_dir
        self.label_info = open(os.path.join(list_dir, "config.yaml")).readlines()
        self.prompt = prompt

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_name = self.sample_list[idx].strip('\n')
        img_path = os.path.join(self.data_dir, img_name)

        image = cv2.imread(img_path)
        label = int(img_name.split("/")[0])

        sample = {'image': image/255.0, 'label': np.zeros(image.shape[:2])}
        if self.transform:
            sample = self.transform(sample)
        sample['label'] = torch.from_numpy(np.array(label))
        sample['case_name'] = self.sample_list[idx].strip('\n')
        if self.prompt:
            dataset_name = img_path.split("/")[-3]
            sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
            sample['nature_prompt'] = nature_prompt_one_hot_dict[nature_prompt_dict[dataset_name]]
            sample['position_prompt'] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        return sample
