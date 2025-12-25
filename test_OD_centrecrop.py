"""
MIT License

Copyright (c) 2025 Zhenyi Zhao
Copyright (c) University of Dundee
"""


import argparse
import monai
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from model.baselines.dunet import DUNet
from utils.transform import validation_transform
from utils.datasets import TestDataset


def get_args_parser():
    parser = argparse.ArgumentParser('test pre-trained DUNet for optic disc segmentation', add_help=False)
    
    # Dataset parameters
    parser.add_argument('--label_n_cls', default=3, type=int, 
                        help='the number of label classes')
    parser.add_argument('--test_image_path', default='./datasets/Vampire/testing/ROIs/images', type=str,
                        help='testing image folder path')
    parser.add_argument('--test_mask_path', default='./datasets/Vampire/testing/ROIs/masks', type=str,
                        help='testing mask folder path')
    parser.add_argument('--image_size', default=256, type=int, 
                        help='input image size')
        
    # Model parameters
    parser.add_argument('--model_selection', default='baseline_DUNet', type=str,
                        help='fixed selection: baseline_DUNet')
    parser.add_argument('--checkpoint_path', default='./DUNet_OD_CentreCrop_pretrained.pth', type=str,
                        help='weights path')
    parser.add_argument('--output_channel', default=2, type=int, 
                        help='the number of model output channels')  
    return parser


def main(args):
    test_image_path = args.test_image_path
    test_mask_path = args.test_mask_path
    image_size = args.image_size
    checkpoint_path = args.checkpoint_path
    output_channel = args.output_channel
    label_n_cls = args.label_n_cls
   
    # build model architecture
    if args.model_selection == 'baseline_DUNet':        
        model = DUNet(input_channel=3, output_channel=output_channel, base_channel=32)
    else:
        raise ValueError('Please select correct model')
    
    # load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model, strict=True)
    
    # load testing dataset
    test_transform = validation_transform(image_size)
    test_dataset = TestDataset(image_root_path = test_image_path, 
                               mask_root_path = test_mask_path, 
                               transform = test_transform,
                               n_cls = label_n_cls,
                              )
    
    # metrics
    od_dice_metrics = monai.metrics.DiceMetric(include_background=True)    
    
    # test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i in test_dataset:
            x, y_true, image_original, filename = i[0].unsqueeze(dim=0), i[1].unsqueeze(dim=0), i[2], i[3]
            y_true = F.one_hot(y_true.squeeze(dim=1), num_classes=label_n_cls).permute(0,3,1,2) 
            x, y_true = x.to(device), y_true.to(device)               
            y_pred = model(x)
            y_pred = F.interpolate(y_pred, size=(y_true.size(2), y_true.size(3)), mode='bicubic')
            y_pred = torch.argmax(torch.softmax(y_pred,1), dim=1)                    
            y_pred = F.one_hot(y_pred, num_classes=output_channel).permute(0,3,1,2)
            
            # OD only                     
            if label_n_cls == 2:
                y_pred = y_pred
                y_true = y_true
            elif label_n_cls == 3:
                y_pred = y_pred
                y_true[:,1,:,:] = y_true[:,1,:,:] + y_true[:,2,:,:]
            else:
                raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
            od_dice_metrics(y_pred[:,1,:,:].unsqueeze(dim=1), y_true[:,1,:,:].unsqueeze(dim=1))
                
        od_dice_mean = od_dice_metrics.aggregate().item()      
        print('*')
        print('*')
        print(f'Number of test images: {len(test_dataset)}')
        print('Best epoch is', checkpoint['epoch'])            
        print('OD mean Dice is', od_dice_mean)
        print('*')
        print('*')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)