# augmentation.py
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
composed_transforms = []
def get_augmentation(augmentation_list):
    for augmentation_type in augmentation_list:
        if augmentation_type == 'blur':
            composed_transforms.append(transforms.GaussianBlur(kernel_size=3))
        elif augmentation_type == 'brightness':
            brightness_factor = 0.3
            composed_transforms.append(transforms.ColorJitter(brightness=brightness_factor))
        elif augmentation_type == 'colorjitter':
            composed_transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
        elif augmentation_type == 'contrast':
            contrast_factor = 0.5
            composed_transforms.append(transforms.ColorJitter(contrast=contrast_factor))
        elif augmentation_type == 'random_affine':
            composed_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)))
        elif augmentation_type == 'rotation':
            angle = 10
            composed_transforms.append(transforms.RandomRotation(degrees=angle))
        elif augmentation_type == 'saturation':
            saturation_factor = 0.5
            composed_transforms.append(transforms.ColorJitter(saturation=saturation_factor))
        elif augmentation_type == 'shear':
            shear = 10
            composed_transforms.append(transforms.RandomAffine(degrees=0, shear=shear))
        elif augmentation_type == 'single_channel':
            composed_transforms.append(transforms.Grayscale(num_output_channels=1))
        elif augmentation_type == 'random_transform':
            composed_transforms.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.5))
        elif augmentation_type == 'elastic_transform':
            composed_transforms.append(transforms.ElasticTransform(alpha=50.0, sigma=5.0))
        elif augmentation_type == 'random_invert':
            composed_transforms.append(transforms.RandomInvert(p=0.2))
        elif augmentation_type == 'random_posterize':
            composed_transforms.append(transforms.RandomPosterize(bits=2, p=0.5))
        elif augmentation_type == 'random_solarize':
            composed_transforms.append(transforms.RandomSolarize(threshold=150, p=0.3))
        elif augmentation_type == 'random_sharpeness':
            composed_transforms.append(transforms.RandomAdjustSharpness(sharpness_factor=3))
        elif augmentation_type == 'random_autocontrast':
            composed_transforms.append(transforms.RandomAutocontrast(p=0.4))
        elif augmentation_type == 'random_equalize':
            composed_transforms.append(transforms.RandomEqualize(p=0.4))
    
    composed_transforms.append(transforms.Grayscale(num_output_channels=3))
    composed_transforms.append(transforms.ToTensor())
    composed_transforms.append(transforms.Normalize((0.1307,), (0.3081,)))
    print(composed_transforms)
    return transforms.Compose(composed_transforms)
    

"""elif augmentation_type == 'histogram_equalization':
            class HistogramEqualization(object):
                def __call__(self, img):
                    np_img = np.array(img)
                    eq_img = np.zeros_like(np_img)
                    for c in range(3):
                        eq_img[:, :, c] = cv2.equalizeHist(np_img[:, :, c])
                    return Image.fromarray(eq_img)
            composed_transforms.append(HistogramEqualization())"""