#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from models.center_net import DownsampleNetBase, CenterNet
from models.u_net import UNet2d
from models.head_net import ObjectDetectionNetBase, HeatmapNetBase, LocalOffsetNetBase, SizeNetBase
from criterions.multi_criterion import ObjectDetectionLoss
from utils.iou import IoU

RECTANGLE_COLOR = (0,0,255)
PIXEL = 2
FONT_SCALE = 2

parser = argparse.ArgumentParser("Demo of Center-Net")

parser.add_argument('--H', type=int, default=256, help='Height of image')
parser.add_argument('--W', type=int, default=256, help='Width of image')
parser.add_argument('--K', type=int, default=5, help='Kernel size')

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

parser.add_argument('--n_candidate', type=int, default=10, help='Number of bbox candidates')
parser.add_argument('--confidence_threshold', type=float, default=0.9, help='Confidence')
parser.add_argument('--iou_threshold', type=float, default=0.1, help='IoU')

parser.add_argument('--model_path', type=str, default=None, help='Path for model')


def main(args):
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
    
    head_modules = {
        'heatmap': HeatmapNetBase(channels_backbone[0]),
        'local_offset': LocalOffsetNetBase(channels_backbone[0]),
        'size': SizeNetBase(channels_backbone[0])
    }
    
    head_net = ObjectDetectionNetBase(head_modules=head_modules)

    model = CenterNet(downsample_net, backbone, head_net)
    print(model)
    print("# Parameters:", model.num_parameters)
    
    model_path = args.model_path
    
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(package['state_dict'])
    
    play_realtime(model, H=args.H, W=args.W, R=args.R, K=args.K, n_candidate=args.n_candidate, confidence_threshold=args.confidence_threshold, iou_threshold=args.iou_threshold)
    # play_offline(model, H=args.H, W=args.W, R=args.R, K=args.K, image_path='./sample.jpg', n_candidate=args.n_candidate, confidence_threshold=args.confidence_threshold, iou_threshold=args.iou_threshold)
    

def play_realtime(model, H, W, R, K, n_candidate=10, confidence_threshold=0.9, iou_threshold=0.1, camera_id=0, delay=1):
    model.eval()

    video_capture = cv2.VideoCapture(camera_id)
    
    ret, original_image = video_capture.read()
    
    if ret:
        original_height, original_width, _ = original_image.shape
    else:
        video_capture.release()
        cv2.destroyAllWindows()

        return
        
    while video_capture.isOpened():
        ret, original_image = video_capture.read()
        
        if ret:
            image = cv2.resize(original_image, (W, H))
            image = image[...,::-1].transpose(2,0,1)
            image = image / 255.0
            
            with torch.no_grad():
                input = torch.Tensor(image).float().unsqueeze(dim=0)
                
                output = model(input)
                
                estimated_bboxes = IoU.estimate_bboxes(output, R=R, kernel_size=K, n_candidate=n_candidate, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold)
            
            image = draw_bbox(original_image, estimated_bboxes, H, W)
            
            cv2.imshow("Estimated bounding box", image)
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        else:
            break
            
    video_capture.release()
    cv2.destroyAllWindows()
    
def play_offline(model, H, W, R, K, image_path, n_candidate=10, confidence_threshold=0.9, iou_threshold=0.1):
    model.eval()
    
    original_image = cv2.imread(image_path)

    original_height, original_width, _ = original_image.shape
    
    image = cv2.resize(original_image, (W, H))
    image = image[...,::-1].transpose(2,0,1)
    image = image / 255.0
    
    with torch.no_grad():
        input = torch.Tensor(image).float().unsqueeze(dim=0)
        
        output = model(input)
        
        estimated_bboxes = IoU.estimate_bboxes(output, R=R, kernel_size=K, n_candidate=n_candidate, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold)
    
    image = draw_bbox(original_image, estimated_bboxes, H, W)
    
    plt.imshow(image[...,::-1])
    plt.savefig('./tmp.png', bbox_inches='tight')
    
    
def draw_bbox(original_image, estimated_bbox, H, W, label="Face"):
    image = np.copy(original_image)
    original_height, original_width, _ = image.shape
    ratio_y, ratio_x = original_height / H, original_width / W

    for bbox in estimated_bbox:
        confidence = bbox['confidence']*100
        p1 = int(ratio_x*bbox['box2d']['x1']), int(ratio_y*bbox['box2d']['y1'])
        p2 = int(ratio_x*bbox['box2d']['x2']), int(ratio_y*bbox['box2d']['y2'])
        
        cv2.rectangle(image, p1, p2, RECTANGLE_COLOR, PIXEL)
        cv2.putText(image, label + " : {:.2f} %".format(confidence), p1, cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, (225,255,255), PIXEL)
        
    return image

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    main(args)


