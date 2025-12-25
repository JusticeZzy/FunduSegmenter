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

from model.fundusegmenter import FunduSegmenter
from model.modules.vit import VisionTransformer
from model.modules.segmenter_decoder import SegmenterDecoder
from model.modules.adapters import PreAdapter, PostAdapter
from utils.transform import validation_transform
from utils.datasets import TestDataset


def get_args_parser():
    parser = argparse.ArgumentParser('test FunduSegmenter for OD/OC segmentation on original images', add_help=False)
    
    # Dataset parameters
    parser.add_argument('--label_n_cls', default=3, type=int, 
                        help='the number of label classes')
    parser.add_argument('--test_image_path', default='./datasets/Vampire/testing/images', type=str,
                        help='testing image folder path')
    parser.add_argument('--test_mask_path', default='./datasets/Vampire/testing/masks', type=str,
                        help='testing mask folder path')
    parser.add_argument('--image_size', default=256, type=int, 
                        help='input image size')
        
    # Model parameters
    parser.add_argument('--model_selection', default='FunduSegmenter', type=str,
                        help='select model from FunduSegmenter, baseline_RETFoundSegmenter, baseline_DUNet or baseline_TransUNet')
    parser.add_argument('--checkpoint_path', default='./results/saved_weights/best_weights.pth', type=str,
                        help='weights path')
    parser.add_argument('--output_channel', default=3, type=int, 
                        help='the number of model output channels')
    parser.add_argument('--norm', default='bn', type=str,
                        help='select normalization for adapters, bn or gn')

    # Output parameters
    parser.add_argument('--segmentation_map_dir', default='./results/segmentation_map/map', type=str,
                        help='directory for saving segmentation map')
    parser.add_argument('--map_contours_dir', default='./results/segmentation_map/map_contours', type=str,
                        help='directory for saving segmentation map with contours')    
    return parser


def main(args):
    test_image_path = args.test_image_path
    test_mask_path = args.test_mask_path
    image_size = args.image_size
    checkpoint_path = args.checkpoint_path
    output_channel = args.output_channel
    label_n_cls = args.label_n_cls
    norm = args.norm
    os.makedirs(args.segmentation_map_dir, exist_ok=True)
    os.makedirs(args.map_contours_dir, exist_ok=True)
    
    # build model architecture
    if args.model_selection == 'FunduSegmenter':
        pre_adapter = PreAdapter(input_channel=3, mid_channel=64, norm=norm)
        encoder = VisionTransformer(image_size=(224,224),
                                    patch_size=16,
                                    n_layers=24,
                                    d_model=1024,
                                    d_ff=4096,
                                    n_heads=16,
                                    n_cls=1000,
                                    dropout=0.0,
                                    drop_path_rate=0.0,
                                    distilled=False,
                                    channels=3,
                                   )

        decoder = SegmenterDecoder(n_cls=output_channel,
                                   patch_size=16,
                                   d_encoder=1024,
                                   n_layers=2,
                                   n_heads=16,
                                   d_model=1024,
                                   d_ff=4096,
                                   drop_path_rate=0.0,
                                   dropout=0.1,
                                   )
        post_adapter = PostAdapter(input_channel=output_channel, mid_channel=64, output_channel=output_channel, norm=norm)
        model = FunduSegmenter(pre_adapter=pre_adapter, encoder=encoder, decoder=decoder, post_adapter=post_adapter)
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
    od_hd_metrics = monai.metrics.HausdorffDistanceMetric(include_background=True, percentile=95, directed=True)
    od_asd_metrics = monai.metrics.SurfaceDistanceMetric(include_background=True)
    
    oc_dice_metrics = monai.metrics.DiceMetric(include_background=True)    
    oc_hd_metrics = monai.metrics.HausdorffDistanceMetric(include_background=True, percentile=95, directed=True)
    oc_asd_metrics = monai.metrics.SurfaceDistanceMetric(include_background=True)
    
    # test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    od_dice_best = 0
    od_dice_worst = 1
    od_best_filename = 0
    od_worst_filename = 0
    oc_dice_best = 0
    oc_dice_worst = 1
    oc_best_filename = 0
    oc_worst_filename = 0
    mean_best = 0
    mean_worst = 1
    mean_best_filename = 0
    mean_worst_filename = 0
    with torch.no_grad():
        for i in test_dataset:
            x, y_true, image_original, filename = i[0].unsqueeze(dim=0), i[1].unsqueeze(dim=0), i[2], i[3]
            y_true = F.one_hot(y_true.squeeze(dim=1), num_classes=label_n_cls).permute(0,3,1,2) 
            x, y_true = x.to(device), y_true.to(device)   
            
            y_pred = model(x)
            y_pred = F.interpolate(y_pred, size=(y_true.size(2), y_true.size(3)), mode='bicubic')
            y_pred = torch.argmax(torch.softmax(y_pred,1), dim=1)           
            y_pred_save = y_pred

            y_pred_save = np.asarray(y_pred_save.to('cpu').permute(1,2,0)).astype(np.uint8)
           
            if output_channel == 2:
                y_pred_save[y_pred_save==1] = 255
                y_pred_save[y_pred_save==2] = 255
                cv2.imwrite(os.path.join(args.segmentation_map_dir, filename[:-4]+'.png'), y_pred_save, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            elif output_channel == 3:
                if label_n_cls == 2:
                    y_pred_save[y_pred_save==1] = 255
                    y_pred_save[y_pred_save==2] = 255
                    cv2.imwrite(os.path.join(args.segmentation_map_dir, filename[:-4]+'.png'), y_pred_save, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                elif label_n_cls == 3:
                    y_pred_save[y_pred_save==0] = 255
                    y_pred_save[y_pred_save==1] = 128
                    y_pred_save[y_pred_save==2] = 0
                    cv2.imwrite(os.path.join(args.segmentation_map_dir, filename[:-4]+'.png'), y_pred_save, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
            else:
                raise ValueError('Please select correct output channels')
         
            y_pred = F.one_hot(y_pred, num_classes=output_channel).permute(0,3,1,2)
            # OD only
            if output_channel == 2:                       
                if label_n_cls == 2:
                    y_pred = y_pred
                    y_true = y_true
                elif label_n_cls == 3:
                    y_pred = y_pred
                    y_true[:,1,:,:] = y_true[:,1,:,:] + y_true[:,2,:,:]
                else:
                    raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
            # OD/OC    
            elif output_channel == 3:
                if label_n_cls == 2:
                    y_pred[:,1,:,:] = y_pred[:,1,:,:] + y_pred[:,2,:,:]
                    y_true = y_true
                elif label_n_cls == 3: 
                    y_pred[:,1,:,:] = y_pred[:,1,:,:] + y_pred[:,2,:,:]
                    y_true[:,1,:,:] = y_true[:,1,:,:] + y_true[:,2,:,:]                
                else:
                    raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
            else:
                raise ValueError('Please select correct output channels')

            od_dice_metrics(y_pred[:,1,:,:].unsqueeze(dim=1), y_true[:,1,:,:].unsqueeze(dim=1))
            od_hd_metrics(y_pred[:,1,:,:].unsqueeze(dim=1), y_true[:,1,:,:].unsqueeze(dim=1))
            od_asd_metrics(y_pred[:,1,:,:].unsqueeze(dim=1), y_true[:,1,:,:].unsqueeze(dim=1))
            od_dice_single = float(od_dice_metrics(y_pred[:,1,:,:].unsqueeze(dim=1), y_true[:,1,:,:].unsqueeze(dim=1)))
            if od_dice_best < od_dice_single:
                od_dice_best = od_dice_single
                od_best_filename = filename
            else:
                od_dice_best = od_dice_best
                od_best_filename = od_best_filename
                                
            if od_dice_worst > od_dice_single:
                od_dice_worst = od_dice_single
                od_worst_filename = filename
            else:
                od_dice_worst = od_dice_worst
                od_worst_filename = od_worst_filename

            od_true = np.asarray(y_true[:,1,:,:].to('cpu').permute(1,2,0)).astype(np.uint8)
            od_pred = np.asarray(y_pred[:,1,:,:].to('cpu').permute(1,2,0)).astype(np.uint8)

            contours_od_true, _ = cv2.findContours(od_true, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_od_pred, _ = cv2.findContours(od_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_countour_od_true = cv2.drawContours(image_original.copy(), 
                                                      contours=contours_od_true, 
                                                      contourIdx=-1, 
                                                      color=(255,0,0), 
                                                      thickness=round(y_true.size(2)/800),
                                                      )
            final = cv2.drawContours(image_countour_od_true.copy(), 
                                     contours=contours_od_pred, 
                                     contourIdx=-1, 
                                     color=(0,255,0), 
                                     thickness=round(y_true.size(2)/800),
                                     )            
           
            if output_channel == 3 and label_n_cls == 3:            
                oc_dice_metrics(y_pred[:,2,:,:].unsqueeze(dim=1), y_true[:,2,:,:].unsqueeze(dim=1))
                oc_hd_metrics(y_pred[:,2,:,:].unsqueeze(dim=1), y_true[:,2,:,:].unsqueeze(dim=1))
                oc_asd_metrics(y_pred[:,2,:,:].unsqueeze(dim=1), y_true[:,2,:,:].unsqueeze(dim=1))
                oc_dice_single = float(oc_dice_metrics(y_pred[:,2,:,:].unsqueeze(dim=1), y_true[:,2,:,:].unsqueeze(dim=1)))
                mean = (od_dice_single+oc_dice_single)/2
                if oc_dice_best < oc_dice_single:
                    oc_dice_best = oc_dice_single
                    oc_best_filename = filename
                else:
                    oc_dice_best = oc_dice_best
                    oc_best_filename = oc_best_filename
                                
                if oc_dice_worst > oc_dice_single:
                    oc_dice_worst = oc_dice_single
                    oc_worst_filename = filename
                else:
                    oc_dice_worst = oc_dice_worst
                    oc_worst_filename = oc_worst_filename

                if mean_best < mean:
                    mean_best = mean
                    mean_best_filename = filename
                else:
                    mean_best = mean_best
                    mean_best_filename = mean_best_filename
                                
                if mean_worst > mean:
                    mean_worst = mean
                    mean_worst_filename = filename
                else:
                    mean_worst = mean_worst
                    mean_worst_filename = mean_worst_filename

                oc_true = np.asarray(y_true[:,2,:,:].to('cpu').permute(1,2,0)).astype(np.uint8)
                oc_pred = np.asarray(y_pred[:,2,:,:].to('cpu').permute(1,2,0)).astype(np.uint8)
                contours_oc_true, _ = cv2.findContours(oc_true, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_oc_pred, _ = cv2.findContours(oc_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                image_countour_oc_true = cv2.drawContours(final.copy(), 
                                                          contours=contours_oc_true, 
                                                          contourIdx=-1, 
                                                          color=(255,0,0), 
                                                          thickness=round(y_true.size(2)/800),
                                                         )
                final = cv2.drawContours(image_countour_oc_true.copy(), 
                                         contours=contours_oc_pred, 
                                         contourIdx=-1, 
                                         color=(0,0,255), 
                                         thickness=round(y_true.size(2)/800),
                                         )
            final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.map_contours_dir, filename[:-4]+'.png'), final, [cv2.IMWRITE_PNG_COMPRESSION, 0])           
        print('*')
        print('*')
        print(f'Number of test images: {len(test_dataset)}')
        print('*')
        print('*')
        print('Best epoch is', checkpoint['epoch'])
        print('*')
        print('*')
        
        od_dice_mean = od_dice_metrics.aggregate().item()   
        od_hd_mean = od_hd_metrics.aggregate().item()   
        od_asd_mean = od_asd_metrics.aggregate().item() 
        print(f'Best OD dice is {od_dice_best} on {od_best_filename}')
        print(f'Worst OD dice is {od_dice_worst} on {od_worst_filename}')
        print('OD mean Dice is', od_dice_mean)
        print('OD mean 95% HD is', od_hd_mean)
        print('OD mean ASD is', od_asd_mean)
        print('*')
        print('*')
        if output_channel == 3 and label_n_cls == 3:
            oc_dice_mean = oc_dice_metrics.aggregate().item()   
            oc_hd_mean = oc_hd_metrics.aggregate().item()   
            oc_asd_mean = oc_asd_metrics.aggregate().item() 
            print(f'Best OC dice is {oc_dice_best} on {oc_best_filename}')
            print(f'Worst OC dice is {oc_dice_worst} on {oc_worst_filename}')
            print('OC mean Dice is', oc_dice_mean)
            print('OC mean 95% HD is', oc_hd_mean)
            print('OC mean ASD is', oc_asd_mean)
            print('*')
            print('*')
            print(f'Best mean dice is {mean_best} on {mean_best_filename}')
            print(f'Worst mean dice is {mean_worst} on {mean_worst_filename}')
            print('*')
            print('*')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)