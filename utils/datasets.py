"""
MIT License

Copyright (c) 2025 Zhenyi Zhao
Copyright (c) University of Dundee
"""


import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class DatasetFromRoot(Dataset):
    def __init__(self, image_root_path, mask_root_path, transform, n_cls):

        self.image_all = []
        self.mask_all = []
        
        image_list = os.listdir(image_root_path)
        image_list.sort()       
        for i in image_list:
            image_path = image_root_path + os.sep + i
            image = np.asarray(Image.open(image_path))
            self.image_all.append(image)
        
        mask_list = os.listdir(mask_root_path)
        mask_list.sort()       
        for j in mask_list:
            mask_path = mask_root_path + os.sep + j
            mask = np.asarray(Image.open(mask_path)).astype(int)
            if n_cls == 2:
                mask[mask == 255] = 1
            elif n_cls == 3:
                mask[mask == 0] = 2
                mask[mask == 128] = 1
                mask[mask == 255] = 0
            else:
                raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
                
            self.mask_all.append(mask)
        
        self.transform_image, self.transform_both = transform
        
    def __len__(self):
        return len(self.image_all)

    def __getitem__(self, idx):
        image = self.image_all[idx] 
        mask = self.mask_all[idx]
        
        transformed_image = self.transform_image(image=image)
        image = transformed_image['image']
        
        transformed_both = self.transform_both(image=image, mask=mask)
        image = transformed_both['image']
        mask = transformed_both['mask']
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        else:
            mask = mask
        mask = mask.unsqueeze(0)
        mask = mask.long()
        return image, mask


class DatasetFromList(Dataset):
    def __init__(self, image_list, mask_list, transform, n_cls):

        self.image_all = []
        self.mask_all = []
        
        image_list = image_list
        image_list.sort()       
        for image_path in image_list:
            image = np.asarray(Image.open(image_path))
            self.image_all.append(image)
        
        mask_list = mask_list
        mask_list.sort()       
        for mask_path in mask_list:
            mask = np.asarray(Image.open(mask_path)).astype(int)
            if n_cls == 2:
                mask[mask == 255] = 1
            elif n_cls == 3:
                mask[mask == 0] = 2
                mask[mask == 128] = 1
                mask[mask == 255] = 0
            else:
                raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
                
            self.mask_all.append(mask)
        
        self.transform_image, self.transform_both = transform
        
    def __len__(self):
        return len(self.image_all)

    def __getitem__(self, idx):
        image = self.image_all[idx] 
        mask = self.mask_all[idx]
        
        transformed_image = self.transform_image(image=image)
        image = transformed_image['image']
        
        transformed_both = self.transform_both(image=image, mask=mask)
        image = transformed_both['image']
        mask = transformed_both['mask']
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        else:
            mask = mask
        mask = mask.unsqueeze(0)
        mask = mask.long()
        return image, mask


class TestDataset(Dataset):
    def __init__(self, image_root_path, mask_root_path, transform, n_cls):

        self.image_all = []
        self.mask_all = []
        self.filename_list = []
        
        image_list = os.listdir(image_root_path)
        image_list.sort()       
        for i in image_list:
            image_path = image_root_path + os.sep + i
            image = np.asarray(Image.open(image_path))
            self.image_all.append(image)
            self.filename_list.append(i)
        
        mask_list = os.listdir(mask_root_path)
        mask_list.sort()       
        for j in mask_list:
            mask_path = mask_root_path + os.sep + j
            mask = np.asarray(Image.open(mask_path)).astype(int)
            if n_cls == 2:
                mask[mask == 255] = 1
            elif n_cls == 3:
                mask[mask == 0] = 2
                mask[mask == 128] = 1
                mask[mask == 255] = 0
            else:
                raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
                
            self.mask_all.append(mask)
        
        self.transform_image, self.transform_both = transform
        
    def __len__(self):
        return len(self.image_all)

    def __getitem__(self, idx):
        image = self.image_all[idx] 
        mask = self.mask_all[idx]
        image_original = self.image_all[idx] 
        filename = self.filename_list[idx]
        
        transformed_image = self.transform_image(image=image)
        image = transformed_image['image']
        
        transformed_both = self.transform_both(image=image, mask=mask)
        image = transformed_both['image']
        mask = transformed_both['mask']
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        else:
            mask = mask
        mask = mask.unsqueeze(0)
        mask = mask.long()
        return image, mask, image_original, filename


class ReconstructTestDataset(Dataset):
    def __init__(self, image_root_path, cropped_mask_root_path, original_mask_root_path, centre_file_path, is_idrid, transform, n_cls):

        self.image_all = []
        self.cropped_mask_all = []
        self.original_mask_all = []
        self.filename_list = []
        self.centre_list = []

        with open(centre_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        lines_list = []
        for line in lines[1:]:
           parts = line.strip().split(',')
           lines_list.append([parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])])
        
        image_list = os.listdir(image_root_path)
        image_list.sort()       
        for i in image_list:
            image_path = image_root_path + os.sep + i
            image = np.asarray(Image.open(image_path))
            self.image_all.append(image)
            self.filename_list.append(i)
            for index, item in enumerate(lines_list):
                if item[0][:-4] == i[:-4]:
                    self.centre_list.append([(item[1], item[2]), (item[3], item[4])])
        
        cropped_mask_list = os.listdir(cropped_mask_root_path)
        cropped_mask_list.sort()       
        for j in cropped_mask_list:
            cropped_mask_path = cropped_mask_root_path + os.sep + j
            cropped_mask = np.asarray(Image.open(cropped_mask_path)).astype(int)
            if n_cls == 2:
                cropped_mask[cropped_mask == 255] = 1
            elif n_cls == 3:
                cropped_mask[cropped_mask == 0] = 2
                cropped_mask[cropped_mask == 128] = 1
                cropped_mask[cropped_mask == 255] = 0
            else:
                raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
                
            self.cropped_mask_all.append(cropped_mask)

        original_mask_list = os.listdir(original_mask_root_path)
        original_mask_list.sort()       
        for j in original_mask_list:
            original_mask_path = original_mask_root_path + os.sep + j
            original_mask = np.asarray(Image.open(original_mask_path)).astype(int)
            if n_cls == 2:
                original_mask = original_mask
            elif n_cls == 3:
                original_mask[original_mask == 0] = 2
                original_mask[original_mask == 128] = 1
                original_mask[original_mask == 255] = 0
            else:
                raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
                
            self.original_mask_all.append(original_mask)
        
        self.transform_image, self.transform_both = transform
        
    def __len__(self):
        return len(self.image_all)

    def __getitem__(self, idx):
        image = self.image_all[idx] 
        cropped_mask = self.cropped_mask_all[idx]
        original_mask = self.original_mask_all[idx]
        image_original = self.image_all[idx] 
        filename = self.filename_list[idx]
        centre = self.centre_list[idx]
        
        transformed_image = self.transform_image(image=image)
        image = transformed_image['image']
        
        transformed_both = self.transform_both(image=image, mask=original_mask)
        image = transformed_both['image']
        original_mask = transformed_both['mask']
        original_mask = original_mask.unsqueeze(0)
        original_mask = original_mask.long()
        return image, original_mask, image_original, cropped_mask, filename, centre


class CropDatasetNoLabel(Dataset):
    def __init__(self, image_root_path, transform):

        self.image_all = []
        self.filename_list = []
        
        image_list = os.listdir(image_root_path)
        image_list.sort()       
        for i in image_list:
            image_path = image_root_path + os.sep + i
            image = np.asarray(Image.open(image_path))
            self.image_all.append(image)
            self.filename_list.append(i)
        
        self.transform_image, self.transform_both = transform
        
    def __len__(self):
        return len(self.image_all)

    def __getitem__(self, idx):
        image = self.image_all[idx] 
        image_original = self.image_all[idx] 
        filename = self.filename_list[idx]
        
        transformed_image = self.transform_image(image=image)
        image = transformed_image['image']
        
        transformed_both = self.transform_both(image=image)
        image = transformed_both['image']
        return image, image_original, filename


class CropDatasetWithLabel(Dataset):
    def __init__(self, image_root_path, mask_root_path, transform):

        self.image_all = []
        self.mask_all = []
        self.filename_list = []
        
        image_list = os.listdir(image_root_path)
        image_list.sort()       
        for i in image_list:
            image_path = image_root_path + os.sep + i
            image = np.asarray(Image.open(image_path))
            self.image_all.append(image)
            self.filename_list.append(i)
        
        mask_list = os.listdir(mask_root_path)
        mask_list.sort()       
        for j in mask_list:
            mask_path = mask_root_path + os.sep + j
            mask = np.asarray(Image.open(mask_path))               
            self.mask_all.append(mask)
        
        self.transform_image, self.transform_both = transform
        
    def __len__(self):
        return len(self.image_all)

    def __getitem__(self, idx):
        image = self.image_all[idx] 
        mask = self.mask_all[idx]
        image_original = self.image_all[idx] 
        filename = self.filename_list[idx]
        
        transformed_image = self.transform_image(image=image)
        image = transformed_image['image']
        
        transformed_both = self.transform_both(image=image)
        image = transformed_both['image']
        return image, mask, image_original, filename