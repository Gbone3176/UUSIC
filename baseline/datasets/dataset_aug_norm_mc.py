import torch
import numpy as np
import random
from scipy.ndimage import zoom
from scipy import ndimage
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


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