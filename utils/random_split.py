"""
MIT License

Copyright (c) 2025 Zhenyi Zhao
Copyright (c) University of Dundee
"""


import os
import random

def random_split(train_image_path, train_mask_path, train_split_rate):
    for image_root_path, _, image_list in os.walk(train_image_path):
        image_list.sort()
    
        image_path = []
        for image in image_list:
            image_path.append(os.path.join(image_root_path, image))
    
    for mask_root_path, _, mask_list in os.walk(train_mask_path):
        mask_list.sort()
    
        mask_path = []
        for mask in mask_list:
            mask_path.append(os.path.join(mask_root_path, mask))

    index = list(range(0, len(image_path)))
    random.shuffle(index)        
    
    train_size = int(len(image_path)*train_split_rate)
    train_image_list = [image_path[i] for i in index[0: train_size]]
    train_mask_list = [mask_path[i] for i in index[0: train_size]]
    val_image_list = [image_path[i] for i in index[train_size: len(image_path)]]
    val_mask_list = [mask_path[i] for i in index[train_size: len(image_path)]]
    return train_image_list, train_mask_list, val_image_list, val_mask_list