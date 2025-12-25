"""
MIT License

Copyright (c) 2025 Zhenyi Zhao
Copyright (c) University of Dundee
"""


import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def no_transform(image_size):
    transform_image = A.Compose([])    
    
    transform_both = A.Compose(
        [A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_AREA, mask_interpolation=cv2.INTER_NEAREST),
         A.ToFloat(),
         A.Normalize(normalization='standard', max_pixel_value=1),
         ToTensorV2(),
        ])
    return transform_image, transform_both

def spatial_transform(image_size):
    transform_image = A.Compose([])    
        
    transform_both = A.Compose(
        [A.Rotate(limit=15, interpolation=cv2.INTER_CUBIC, p=0.5),
         A.VerticalFlip(p=0.5),
         A.HorizontalFlip(p=0.5),
         A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_AREA, mask_interpolation=cv2.INTER_NEAREST),
         A.ToFloat(),
         A.Normalize(normalization='standard', max_pixel_value=1),
         ToTensorV2(),
        ])
    return transform_image, transform_both

def designed_transform(image_size):
    transform_image = A.Compose(
        [A.GaussianBlur(sigma_limit=(0.25, 0.8), p=0.5), 
         A.GaussNoise(std_range=(0, 0.02), per_channel=True, p=0.5), 
         A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
         A.RandomGamma(gamma_limit=(80, 120), p=0.5),
         ])    
    
    transform_both = A.Compose(
        [A.Rotate(limit=15, interpolation=cv2.INTER_CUBIC, p=0.5),
         A.VerticalFlip(p=0.5),
         A.HorizontalFlip(p=0.5),
         A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_AREA, mask_interpolation=cv2.INTER_NEAREST),
         A.ToFloat(),
         A.Normalize(normalization='standard', max_pixel_value=1),
         ToTensorV2(),
        ])
    return transform_image, transform_both

def validation_transform(image_size):    
    transform_image = A.Compose(
        [A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_AREA, mask_interpolation=cv2.INTER_NEAREST),
         A.ToFloat(),
         A.Normalize(normalization='standard', max_pixel_value=1),
        ])    
    
    transform_both = A.Compose(
        [ToTensorV2(),
        ],
        is_check_shapes=False,
        )
    return transform_image, transform_both