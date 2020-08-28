#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import EvalDataset, EvalDataLoader
from models.center_net import DownsampleNetBase, CenterNet
from models.unet import UNet2d
from models.head_net import ObjectDetectionNetBase, HeatmapNetBase, LocalOffsetNetBase, SizeNetBase
from criterions.multi_criterion import ObjectDetectionLoss
from driver import Evaluater

parser = argparse.ArgumentParser("Evaluation of Center-Net")

parser.add_argument('--test_image_root', type=str, default=None, help='Image dataset for test root directory')
parser.add_argument('--test_path', type=str, default=None, help='Annotation path of test data')

parser.add_argument('--H', type=int, default=256, help='Height of image')
parser.add_argument('--W', type=int, default=256, help='Width of image')
parser.add_argument('--K', type=int, default=5, help='Kernel size')
parser.add_argument('--n_candidate', type=int, default=10, help='Number of candidates')
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

parser.add_argument('--eval_dir', type=str, default='./tmp', help='Evaluation output directory')
parser.add_argument('--model_path', type=str, default=None, help='Path for trained model')

def main(args):
    set_seed(111)

    test_dataset = EvalDataset(args.test_image_root, args.test_path, H=args.H, W=args.W, R=args.R, G=args.G)
    print("Test dataset includes {} images.".format(len(test_dataset)))
    
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
        
    evaluater = Evaluater(model, test_dataset, args)
    evaluater.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    main(args)
