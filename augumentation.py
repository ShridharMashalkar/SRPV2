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

def get_augmentation(augmentation_type):
    if augmentation_type == 'al_withoutda':
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'blur':
        return transforms.Compose([
            transforms.GaussianBlur(kernel_size=3),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'brightness':
        brightness_factor = 0.7
        return transforms.Compose([
            transforms.ColorJitter(brightness=brightness_factor),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'colorjitter':
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'contrast':
        contrast_factor = 0.7
        return transforms.Compose([
            transforms.ColorJitter(contrast=contrast_factor),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'histogram_equalization':
        class HistogramEqualization(object):
            def __call__(self, img):
                np_img = np.array(img)
                eq_img = np.zeros_like(np_img)
                for c in range(3):
                    eq_img[:, :, c] = cv2.equalizeHist(np_img[:, :, c])
                return Image.fromarray(eq_img)

        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            HistogramEqualization(),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'random_affine':
        return transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'rotation':
        angle=10
        return transforms.Compose([
            transforms.RandomRotation(degrees=angle),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'saturation':
        saturation_factor = 0.5
        return transforms.Compose([
            transforms.ColorJitter(saturation=saturation_factor),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'shear':
        shear = 10
        return transforms.Compose([
            transforms.RandomAffine(degrees=0, shear=shear),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'single_channel':
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'random_transform':
        return transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'elastic_transform':
        return transforms.Compose([
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    elif augmentation_type == 'random_invert':
        return transforms.Compose([
        transforms.RandomInvert(p=0.2),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    elif augmentation_type == 'random_posterize':
        return transforms.Compose([
        transforms.RandomPosterize(bits=2, p=0.5),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    elif augmentation_type == 'random_solarize':
        return transforms.Compose([
        transforms.RandomSolarize(threshold=150, p=0.3),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    elif augmentation_type == 'random_sharpeness':
        return transforms.Compose([
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    elif augmentation_type == 'random_autocontrast':
        return transforms.Compose([
        transforms.RandomAutocontrast(p=0.4),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    elif augmentation_type == 'random_equalize':
        return transforms.Compose([
        transforms.RandomEqualize(p=0.4),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    elif augmentation_type == 'brightness50contrast50':
        brightness_factor = 0.5
        contrast_factor = 0.5
        return transforms.Compose([
            transforms.ColorJitter(brightness=brightness_factor,contrast=contrast_factor),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'brightness30contrast50':
        brightness_factor = 0.3
        contrast_factor = 0.5
        return transforms.Compose([
            transforms.ColorJitter(brightness=brightness_factor,contrast=contrast_factor),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'brightness70contrast70':
        brightness_factor = 0.7
        contrast_factor = 0.7
        return transforms.Compose([
            transforms.ColorJitter(brightness=brightness_factor,contrast=contrast_factor),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'brightness50contrast30':
        brightness_factor = 0.5
        contrast_factor = 0.3
        return transforms.Compose([
            transforms.ColorJitter(brightness=brightness_factor,contrast=contrast_factor),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'rotation10saturation50':
        angle=10
        saturation_factor = 0.5
        return transforms.Compose([
            transforms.RandomRotation(degrees=angle),
            transforms.ColorJitter(saturation=saturation_factor),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'random_sharpeness_invert':
        return transforms.Compose([
        transforms.RandomAdjustSharpness(sharpness_factor=4),
        transforms.RandomInvert(p=0.2),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    elif augmentation_type == 'blur_shear':
        shear = 10
        return transforms.Compose([
            transforms.RandomAffine(degrees=0, shear=shear),
            transforms.GaussianBlur(kernel_size=3),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'random_solarize_sharpness':
        return transforms.Compose([
        transforms.RandomSolarize(threshold=150, p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=3),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    elif augmentation_type == 'brightness30contrast50rotation10':
        brightness_factor = 0.3
        contrast_factor = 0.5
        angle=10
        return transforms.Compose([
            transforms.ColorJitter(brightness=brightness_factor,contrast=contrast_factor),
            transforms.RandomRotation(degrees=angle),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'random_autocontrast_affine_equalize':
        return transforms.Compose([
        transforms.RandomAutocontrast(p=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.RandomEqualize(p=0.2),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    elif augmentation_type == 'random_rotation_shear_saturation':
        return transforms.Compose([
          transforms.RandomRotation(degrees=15),  
          transforms.RandomAffine(degrees=0, shear=10),  
          transforms.ColorJitter(saturation=0.5), 
          transforms.Grayscale(num_output_channels=3),
          transforms.ToTensor(),                  
          transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif augmentation_type == 'random_posterize_transform_contrast':
        return transforms.Compose([
        transforms.ColorJitter(contrast=0.7),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomPosterize(bits=2, p=0.5),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    elif augmentation_type == 'brightness50contrast90':
        brightness_factor = 0.5
        contrast_factor = 0.9
        return transforms.Compose([
            transforms.ColorJitter(brightness=brightness_factor,contrast=contrast_factor),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        raise ValueError("Invalid augmentation type")
