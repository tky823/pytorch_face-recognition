#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import TrainDataset, TrainDataLoader
from models.center_net import DownsampleNetBase, CenterNet
from models.u_net import UNet2d
from models.head_net import ObjectDetectionNetBase, HeatmapNetBase, LocalOffsetNetBase, SizeNetBase
from criterions.multi_criterion import ObjectDetectionLoss
from driver import Trainer

parser = argparse.ArgumentParser("Training of Center-Net")

parser.add_argument('--train_image_root', type=str, default=None, help='Image dataset for training root directory')
parser.add_argument('--valid_image_root', type=str, default=None, help='Image dataset for validation root directory')
parser.add_argument('--train_path', type=str, default=None, help='Annotation path of training data')
parser.add_argument('--valid_path', type=str, default=None, help='Annotation path of validation data')

parser.add_argument('--H', type=int, default=256, help='Height of image')
parser.add_argument('--W', type=int, default=256, help='Width of image')
parser.add_argument('--G', type=int, default=5, help='Coefficient of variance')

# Network configration
parser.add_argument('--R', type=int, default=4, help='Resolution ratio')
parser.add_argument('--K_down', type=int, default=3, help='Kernel size of down sampling')
parser.add_argument('--S_down', type=int, default=2, help='Stride size of down sampling')
parser.add_argument('--pool_down', type=str, default='max', choices=['average', 'max'], help='Pooling of down sampling')
parser.add_argument('--channels', type=str, default='[64,128,256,512]', help='Hidden channels of backbone')
parser.add_argument('--K_backbone', type=int, default=3, help='Kernel size of backbone')
parser.add_argument('--S_backbone', type=int, default=2, help='Stride size of backbone')
parser.add_argument('--dilated', type=int, default=0, help='Dilated convolution')
parser.add_argument('--separable', type=int, default=0, help='Depthwise-separable convolution')
parser.add_argument('--pool_backbone', type=str, default='max', choices=['average', 'max'], help='Pooling of backbone')
parser.add_argument('--nonlinear_backbone', type=str, default='relu', choices=['relu', 'sigmoid'], help='Pooling of backbone')

# Criterion
parser.add_argument('--heatmap_loss', type=str, default='bce', choices=['bce', 'focal'], help='Loss for heatmap estimation. bce (binary cross entropy) or focal (focal loss)')
parser.add_argument('--importance', type=str, default='[1.0,1.0,1.0]', help='Weight for heatmap, local offset, size')

parser.add_argument('--batch_size', type=int, default=4, help='Mini batch size')
parser.add_argument('--epochs', type=int, default=100, help='Epoch')

# Optimzer
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

# Saving directory
parser.add_argument('--model_dir', type=str, default='./tmp', help='Model saving directory')
parser.add_argument('--loss_dir', type=str, default='./tmp', help='Loss saving directory')
parser.add_argument('--sample_dir', type=str, default='./tmp', help='Sample output directory')

# Resume training
parser.add_argument('--continue_from', type=str, default=None, help='Resume training')

def main(args):
    set_seed(111)

    train_dataset = TrainDataset(args.train_image_root, args.train_path, H=args.H, W=args.W, R=args.R, G=args.G)
    print("Training dataset includes {} images.".format(len(train_dataset)))
    valid_dataset = TrainDataset(args.valid_image_root, args.valid_path, H=args.H, W=args.W, R=args.R, G=args.G)
    print("Validation dataset includes {} images.".format(len(valid_dataset)))
    
    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    loader['valid'] = TrainDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    C = 3
    channels = args.channels.replace('[','').replace(']','').split(',')
    channels_backbone = [
        int(channel) for channel in channels
    ]
    logR = int(math.log2(args.R))
    
    channels_down = [C]
    
    for r in range(logR//2):
        channel = channels_backbone[0]//(logR//2 - r)
        channels_down.append(channel)

    downsample_net = DownsampleNetBase(channels_down, kernel_size=args.K_down, stride=args.S_down, pool=args.pool_down)
    
    backbone = UNet2d(channels_backbone, kernel_size=args.K_backbone, stride=args.S_backbone, dilated=args.dilated, separable=args.separable, nonlinear_enc=args.nonlinear_backbone, nonlinear_dec=args.nonlinear_backbone)
    
    head_list = ['heatmap', 'local_offset', 'size']
    head_modules = {
        'heatmap': HeatmapNetBase(channels_backbone[0]),
        'local_offset': LocalOffsetNetBase(channels_backbone[0]),
        'size': SizeNetBase(channels_backbone[0])
    }
    
    head_net = ObjectDetectionNetBase(head_modules=head_modules)

    model = CenterNet(downsample_net, backbone, head_net)
    print(model, flush=True)
    print("# Parameters:", model.num_parameters)
    
    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model)
        print("Use CUDA")
    else:
        print("Does NOT use CUDA")
        
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Not support optimizer {}".format(args.optimizer))
        
    # Criterion
    importance = args.importance.replace('[','').replace(']','').split(',')
    importance = {
        head_key: float(importance[idx]) for idx, head_key in enumerate(head_list)
    }
    criterion = ObjectDetectionLoss(importance, args.heatmap_loss)
        
    trainer = Trainer(model, loader, criterion, optimizer, args)
    trainer.run()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    main(args)
