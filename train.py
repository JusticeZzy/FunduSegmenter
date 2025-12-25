"""
MIT License

Copyright (c) 2025 Zhenyi Zhao
Copyright (c) University of Dundee
"""


import argparse
import os
import monai
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.fundusegmenter import FunduSegmenter
from model.modules.vit import VisionTransformer
from model.modules.segmenter_decoder import SegmenterDecoder
from model.modules.adapters import PreAdapter, PostAdapter
from model.utils.pos_embed import interpolate_pos_embed
from model.baselines.retfoundsegmenter import RETFoundSegmenter
from model.baselines.dunet import DUNet
from model.baselines.transunet.networks.vit_seg_modeling import VisionTransformer as TransUNet
from model.baselines.transunet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from utils.set_seed import set_seed
from utils.datasets import DatasetFromRoot, DatasetFromList
from utils.transform import no_transform, spatial_transform, designed_transform, validation_transform
from utils.random_split import random_split


def get_args_parser():
    parser = argparse.ArgumentParser('FunduSegmenter for OD/OC segmentation', add_help=False)
    
    # Training parameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--lr', default=0.001, type=float, 
                       help='learning rate')
    
    # Model parameters
    parser.add_argument('--model_selection', default='FunduSegmenter', type=str,
                        help='select model from FunduSegmenter, baseline_RETFoundSegmenter, baseline_DUNet or baseline_TransUNet')
    parser.add_argument('--encoder_path', default='./RETFound_mae_natureCFP.pth', type=str,
                        help='RETFound weights path')
    parser.add_argument('--transunet_pretrained_path', default='./imagenet21k_R50+ViT-B_16.npz', type=str,
                        help='TransUNet pretrained weight path')
    parser.add_argument('--output_channel', default=3, type=int, 
                        help='the number of model output channels')
    parser.add_argument('--norm', default='bn', type=str,
                        help='select normalization for adapters, bn or gn')
    
    # Dataset parameters
    parser.add_argument('--label_n_cls', default=3, type=int, 
                        help='the number of label classes')
    parser.add_argument('--seed', default=112316, type=int, 
                        help='set seed for reproducibility')
    parser.add_argument('--train_image_path', default='./datasets/Vampire/training/ROIs/images', type=str,
                        help='training image folder path')
    parser.add_argument('--train_mask_path', default='./datasets/Vampire/training/ROIs/masks', type=str,
                        help='training mask folder path')
    parser.add_argument('--separate_val', default=0, type=int,
                       help='choose if validation data is separate, 1 for True, 0 for False')
    parser.add_argument('--val_image_path', default='./datasets/Vampire/validation/ROIs/images', type=str,
                        help='validation image folder path')
    parser.add_argument('--val_mask_path', default='./datasets/Vampire/validation/ROIs/masks', type=str,
                        help='validation mask folder path')
    parser.add_argument('--num_workers', default=1, type=int)    
    parser.add_argument('--image_size', default=256, type=int, 
                        help='input image size')
    parser.add_argument('--transform_mode', default='no_transform', type=str,
                        help='select data transform: no_transform, basic_transform or designed_transform')
    parser.add_argument('--train_split_rate', default=0.8, type=float, 
                       help='training data split rate if no separate validation data')
    
    # Output parameters
    parser.add_argument('--results_base_dir', default='./results', type=str,
                        help='directory for saving results')
    parser.add_argument('--weights_dir', default='./saved_weights', type=str,
                        help='directory for saving weights')
    parser.add_argument('--tb_path', default='./tensorboard', type=str,
                        help='path for saving tensorboard')
    parser.add_argument('--records_figure_path', default='./Records.svg', type=str,
                        help='path for saving loss and dice figure')
    return parser


def main(args):
    set_seed(args.seed)
    image_size = args.image_size
    output_channel = args.output_channel
    norm = args.norm

    ######################################## Model architecture ########################################
    # FunduSegmenter
    if args.model_selection == 'FunduSegmenter':
        # pre-adapter setting
        pre_adapter = PreAdapter(input_channel=3, mid_channel=64, norm=norm)        
        # load RETFound
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
        # load RETFound weights
        checkpoint = torch.load(args.encoder_path, map_location='cpu', weights_only=False)
        checkpoint_encoder = checkpoint['model']
        state_dict = encoder.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_encoder and checkpoint_encoder[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_encoder[k]
        # interpolate position embedding
        interpolate_pos_embed(encoder, checkpoint_encoder)
        # load pre-trained model
        msg = encoder.load_state_dict(checkpoint_encoder, strict=False)    
        # create decoder
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
        # post-adapter setting
        post_adapter = PostAdapter(input_channel=output_channel, mid_channel=64, output_channel=output_channel, norm=norm)    
        # create model
        model = FunduSegmenter(pre_adapter=pre_adapter, encoder=encoder, decoder=decoder, post_adapter=post_adapter)       
        # freeze the encoder
        for p in model.encoder.parameters():
            p.requires_grad = False 
        for block in model.encoder.blocks:
            for p in block.adapter.parameters():
                p.requires_grad = True
        need_grad = [p for p in model.parameters() if p.requires_grad] 
        # calculate parameter amounts and print
        total_parameters = sum(p.numel() for p in model.parameters()) / 1e6
        retfound_parameters = sum(p.numel() for p in model.encoder.parameters()) / 1e6
        preadapter_parameters = sum(p.numel() for p in model.pre_adapter.parameters()) / 1e6
        decoder_parameters = sum(p.numel() for p in model.decoder.parameters()) / 1e6
        postadapter_parameters = sum(p.numel() for p in model.post_adapter.parameters()) / 1e6
        learnable_parameters = sum(p.numel() for p in need_grad) / 1e6
        print("Model = %s" % str(model))
        print('Total number of parameters (M): %.3f' % (total_parameters))
        print('Number of frozen parameters of RETFound (M): %.3f' % (retfound_parameters))
        print('Number of pre_adapter parameters (M): %.3f' % (preadapter_parameters))
        print('Number of decoder parameters (M): %.3f' % (decoder_parameters))
        print('Number of post_adapter parameters (M): %.3f' % (postadapter_parameters))    
        print('Number of learnable parameters (M): %.3f' % (learnable_parameters))
    
    # ablation experiment baseline: RETFoundSegmenter
    elif args.model_selection == 'baseline_RETFoundSegmenter':       
        # load RETFound
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
        # load RETFound weights
        checkpoint = torch.load(args.encoder_path, map_location='cpu', weights_only=False)
        checkpoint_encoder = checkpoint['model']
        state_dict = encoder.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_encoder and checkpoint_encoder[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_encoder[k]
        # interpolate position embedding
        interpolate_pos_embed(encoder, checkpoint_encoder)
        # load pre-trained model
        msg = encoder.load_state_dict(checkpoint_encoder, strict=False)    
        # create decoder
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
        # create model
        if image_size != 224:
            raise ValueError('Input size of baseline_RETFoundSegmenter must be 224*224')
        else:            
            model = RETFoundSegmenter(encoder=encoder, decoder=decoder)       
        # freeze the encoder
        for p in model.encoder.parameters():
            p.requires_grad = False    
        need_grad = [p for p in model.parameters() if p.requires_grad] 
        # calculate parameter amounts and print
        total_parameters = sum(p.numel() for p in model.parameters()) / 1e6
        retfound_parameters = sum(p.numel() for p in model.encoder.parameters()) / 1e6
        decoder_parameters = sum(p.numel() for p in model.decoder.parameters()) / 1e6
        learnable_parameters = sum(p.numel() for p in need_grad) / 1e6
        print("Model = %s" % str(model))
        print('Total number of parameters (M): %.3f' % (total_parameters))
        print('Number of frozen parameters of RETFound (M): %.3f' % (retfound_parameters))
        print('Number of decoder parameters (M): %.3f' % (decoder_parameters))
        print('Number of learnable parameters (M): %.3f' % (learnable_parameters))   

    # DUNet
    elif args.model_selection == 'baseline_DUNet':
        # create decoder
        model = DUNet(input_channel=3, output_channel=output_channel, base_channel=32)
        # calculate parameter amounts and print
        need_grad = [p for p in model.parameters() if p.requires_grad]         
        total_parameters = sum(p.numel() for p in model.parameters()) / 1e6
        learnable_parameters = sum(p.numel() for p in need_grad) / 1e6        
        print("Model = %s" % str(model))
        print('Total number of parameters (M): %.3f' % (total_parameters))
        print('Number of learnable parameters (M): %.3f' % (learnable_parameters)) 

    # TransUNet
    elif args.model_selection == 'baseline_TransUNet':
        # create decoder
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = output_channel
        config_vit.n_skip = 3 
        config_vit.pretrained_path = args.transunet_pretrained_path
        config_vit.patches.grid = (int(image_size / 16), int(image_size / 16))
        model = TransUNet(config_vit, img_size=image_size, num_classes=config_vit.n_classes).cuda()
        # load pre-trained model
        model.load_from(weights=np.load(config_vit.pretrained_path))
        # calculate parameter amounts and print
        need_grad = [p for p in model.parameters() if p.requires_grad]         
        total_parameters = sum(p.numel() for p in model.parameters()) / 1e6
        learnable_parameters = sum(p.numel() for p in need_grad) / 1e6        
        print("Model = %s" % str(model))
        print('Total number of parameters (M): %.3f' % (total_parameters))
        print('Number of learnable parameters (M): %.3f' % (learnable_parameters)) 
    else:
        raise ValueError('Please select correct model')

    ######################################## Datasets ########################################
    # load datasets
    separate_val = args.separate_val
    label_n_cls = args.label_n_cls    
    
    # choose data transform strategy
    transform_mode = args.transform_mode
    val_transform = validation_transform(image_size)
    if transform_mode == 'no_transform':
        train_transform = no_transform(image_size)
    elif transform_mode == 'spatial_transform':
        train_transform = spatial_transform(image_size)
    elif transform_mode == 'designed_transform':
        train_transform = designed_transform(image_size)
    else: 
        raise ValueError('Please select correct transform mode from \'no_transform\', \'spatial_transform\', or \'designed_transform\'.')       
    
    # load data if the validation dataset are separate
    if separate_val == 1:        
        train_dataset = DatasetFromRoot(image_root_path = args.train_image_path, 
                                        mask_root_path = args.train_mask_path, 
                                        transform = train_transform,
                                        n_cls = label_n_cls,
                                       )
        val_dataset = DatasetFromRoot(image_root_path = args.val_image_path, 
                                      mask_root_path = args.val_mask_path, 
                                      transform = val_transform,
                                      n_cls = label_n_cls,
                                     )
    
    # load data if the validation dataset are not separate
    elif separate_val == 0:
        # split train and validation sets
        train_image_list, train_mask_list, val_image_list, val_mask_list = random_split(args.train_image_path, 
                                                                                        args.train_mask_path, 
                                                                                        args.train_split_rate,
                                                                                       )
        train_dataset = DatasetFromList(image_list = train_image_list, 
                                        mask_list = train_mask_list, 
                                        transform = train_transform,
                                        n_cls = label_n_cls,
                                       )
        val_dataset = DatasetFromList(image_list = val_image_list, 
                                      mask_list = val_mask_list, 
                                      transform = val_transform,
                                      n_cls = label_n_cls,
                                     )
    else:
        raise ValueError('Please input 1 for the separate validation dataset, or input 0 for the mixed datasets.')        
    print(f'Number of train images: {len(train_dataset)}')
    print(f'Number of validation images: {len(val_dataset)}')
        
    # dataloader
    batch_size = args.batch_size
    num_workers = args.num_workers    
    loader_train = DataLoader(train_dataset,                           
                              batch_size = batch_size,                           
                              shuffle = True,                           
                              num_workers = num_workers,                        
                              drop_last = True,                         
                             )
    loader_val = DataLoader(val_dataset,                           
                            batch_size = 1,                           
                            shuffle = False,                           
                            num_workers = num_workers,                        
                            drop_last = False,                         
                           )
    
    ######################################## Implementations ########################################
    # loss function
    dice_loss = monai.losses.DiceLoss(include_background=True, squared_pred=False, softmax=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    # epochs
    epochs = args.epochs
    # optimizer
    optimizer = optim.AdamW(need_grad, lr = args.lr, weight_decay=0.001)

    ######################################## Training ########################################   
    # create results directory
    os.makedirs(args.results_base_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_base_dir, args.weights_dir), exist_ok=True)

    # start training
    print(f'If GPU is available: {torch.cuda.is_available()}')    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
    model.to(device)        
    
    loss_train_all = []
    od_dice_all = []
    oc_dice_all = []
    dice_mean_all = []
    loss_train = []
    od_dice = []
    oc_dice = []    
    lr_all = []
    max_dice = -1    
    
    writer = SummaryWriter(os.path.join(args.results_base_dir, args.tb_path))
    print('*')
    print('*')
    print('*')    
    print(f'******** Start training for {epochs} epochs ********')
    print('*')
    print('*')
    print('*')     
    start_time = time.time()
    
    for epoch in tqdm(range(epochs), total = epochs):    
        # training
        model.train()
        for i, (x, y_true) in enumerate(loader_train):                        
            x, y_true = x.to(device), y_true.to(device)      
            y_true = F.one_hot(y_true.squeeze(dim=1), num_classes=label_n_cls).permute(0,3,1,2)  
            y_pred = model(x)           
            optimizer.zero_grad()
            
            if output_channel == 2:
                if label_n_cls == 2:
                    y_true = y_true
                elif label_n_cls == 3:
                    y_true[:,1,:,:] = y_true[:,1,:,:] + y_true[:,2,:,:]
                    y_true = y_true[:,0:2,:,:]
                else:
                    raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
            elif output_channel == 3:
                y_true = y_true
            else:
                raise ValueError('Please select correct output channels')
            
            loss = dice_loss(y_pred, y_true.float()) + ce_loss(y_pred, y_true.float())            
            loss_train.append(loss.item())        
            loss.backward()
            optimizer.step() 
        
        # validation
        model.eval()    
        with torch.no_grad():
            od_dice_metrics = monai.metrics.DiceMetric(include_background=True)
            oc_dice_metrics = monai.metrics.DiceMetric(include_background=True)
            
            for i, (x, y_true) in enumerate(loader_val):                
                x, y_true = x.to(device), y_true.to(device)   
                y_true = F.one_hot(y_true.squeeze(dim=1), num_classes=label_n_cls).permute(0,3,1,2)                  
                y_pred = model(x)
                y_pred = F.interpolate(y_pred, size=(y_true.size(2), y_true.size(3)), mode='bicubic')
                y_pred = torch.argmax(torch.softmax(y_pred,1), dim=1)
                y_pred = F.one_hot(y_pred, num_classes=output_channel).permute(0,3,1,2)
                
                if output_channel == 2:
                    if label_n_cls == 2:
                        od_dice_metrics(y_pred[:,1,:,:].unsqueeze(dim=1), y_true[:,1,:,:].unsqueeze(dim=1))
                        od_dice.append(od_dice_metrics.aggregate().item())
                    elif label_n_cls == 3:
                        y_true[:,1,:,:] = y_true[:,1,:,:] + y_true[:,2,:,:]
                        od_dice_metrics(y_pred[:,1,:,:].unsqueeze(dim=1), y_true[:,1,:,:].unsqueeze(dim=1))
                        od_dice.append(od_dice_metrics.aggregate().item())
                    else:
                        raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
                elif output_channel == 3:
                    y_true[:,1,:,:] = y_true[:,1,:,:] + y_true[:,2,:,:]
                    y_pred[:,1,:,:] = y_pred[:,1,:,:] + y_pred[:,2,:,:]
                    od_dice_metrics(y_pred[:,1,:,:].unsqueeze(dim=1), y_true[:,1,:,:].unsqueeze(dim=1))
                    oc_dice_metrics(y_pred[:,2,:,:].unsqueeze(dim=1), y_true[:,2,:,:].unsqueeze(dim=1))                                       
                    od_dice.append(od_dice_metrics.aggregate().item())
                    oc_dice.append(oc_dice_metrics.aggregate().item())
                else:
                    raise ValueError('Please select correct output channels')
        
        # record current learning rate, training loss and validation dice, and save the best weights
        lr_current= args.lr
        lr_all.append(lr_current)
        print(f'Epoch_{(epoch+1)} learning rate: {lr_current}')        
        
        loss_epoch_train = np.mean(loss_train)
        loss_train_all.append(loss_epoch_train)
        print(f'Epoch_{(epoch+1)} training loss: {loss_epoch_train}')
        loss_train = []  

        if output_channel == 2:
            od_dice_epoch = np.mean(od_dice)
            od_dice_all.append(od_dice_epoch)
            print(f'Epoch_{(epoch+1)} validation OD Dice: {od_dice_epoch}')
            od_dice = []
            
            # save the best model with the most validation dice
            if od_dice_epoch > max_dice:       
                max_dice = od_dice_epoch
                best_epoch = epoch + 1
                best_checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': best_epoch,
                }        
                torch.save(best_checkpoint, os.path.join(args.results_base_dir, args.weights_dir, 'best_weights.pth'))
            
            # tensorboard visualization
            writer.add_scalars('Training Loss and validation dice', {'Loss':loss_epoch_train, 'OD dice':od_dice_epoch}, epoch)
            
        elif output_channel == 3:
            od_dice_epoch = np.mean(od_dice)
            od_dice_all.append(od_dice_epoch)
            print(f'Epoch_{(epoch+1)} validation OD Dice: {od_dice_epoch}')            
            od_dice = []       
            
            oc_dice_epoch = np.mean(oc_dice)
            oc_dice_all.append(oc_dice_epoch)
            print(f'Epoch_{(epoch+1)} validation OC Dice: {oc_dice_epoch}')            
            oc_dice = []   
            
            dice_mean = np.mean([od_dice_epoch, oc_dice_epoch])
            print(f'Epoch_{(epoch+1)} validation mean Dice: {dice_mean}')
            dice_mean_all.append(dice_mean)
            
            # save the best model with the most mean validation dice            
            if dice_mean > max_dice:       
                max_dice = dice_mean
                best_epoch = epoch + 1
                best_checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': best_epoch,
                }        
                torch.save(best_checkpoint, os.path.join(args.results_base_dir, args.weights_dir, 'best_weights.pth'))
            
            # tensorboard visualization
            writer.add_scalars('Training Loss and validation dice', {'Loss':loss_epoch_train, 'OD dice':od_dice_epoch, 'OC dice':oc_dice_epoch, 'Mean dice':dice_mean}, epoch)
        else:
            raise ValueError('Please select correct output channels')       
    writer.close()        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time is {total_time_str}')
    print(f'The best weights are trained in epoch {best_epoch}')

    # training loss and validation dice records visualization
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2    
    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    line1 = ax1.plot(range(0, epochs), loss_train_all, label='Training loss', color='#EF5350', linewidth=2, alpha=0.8)[0]
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=14, fontweight='bold', color='#EF5350')
    ax1.tick_params(axis='y', labelcolor='#EF5350')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax2 = ax1.twinx()
    lines = [line1]
    if output_channel == 2:   
        line2 = ax2.plot(range(0, epochs), od_dice_all, label='OD dice', color='#4CAF50', linewidth=2, alpha=0.8)[0]
        lines.append(line2)
    elif output_channel == 3:    
        line2 = ax2.plot(range(0, epochs), od_dice_all, label='OD dice', color='#FFEB3B', linewidth=2, alpha=0.8)[0]
        line3 = ax2.plot(range(0, epochs), oc_dice_all, label='OC dice', color='#2196F3', linewidth=2, alpha=0.8)[0]
        line4 = ax2.plot(range(0, epochs), dice_mean_all, label='Mean dice', color='#4CAF50', linewidth=2, alpha=0.8)[0]
        lines.extend([line2, line3, line4])
    ax2.set_ylabel('Dice', fontsize=14, fontweight='bold', color='#4CAF50')
    ax2.tick_params(axis='y', labelcolor='#4CAF50')
    plt.title('Training records: Loss and Dice', fontsize=16, fontweight='bold', pad=20)
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=12, framealpha=0.9)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax1.ticklabel_format(axis='both', style='plain', useOffset=False)
    ax2.ticklabel_format(axis='both', style='plain', useOffset=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_base_dir, args.records_figure_path), dpi=1200)
    plt.show()
    plt.close()

    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)