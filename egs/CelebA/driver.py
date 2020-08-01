import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import numpy as np
import matplotlib.pyplot as plt

from utils.utils import draw_loss
from utils.iou import IoU, decode

class Trainer:
    def __init__(self, model, loader, criterion, optimizer, args):
        self.model = model
        
        self.train_loader = loader['train']
        self.valid_loader = loader['valid']
        
        self.criterion = criterion
        
        self.optimizer = optimizer
        
        self._reset(args)
        
    def _reset(self, args):
        self.model_dir = args.model_dir
        self.loss_dir = args.loss_dir
        self.sample_dir = args.sample_dir
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
    
        self.start_epoch = 0
        self.epochs = args.epochs
        
        self.best_loss = float('infinity')
        self.no_improvement = 0
        
        self.train_loss = torch.empty(self.epochs)
        self.valid_loss = torch.empty(self.epochs)
        
        if args.continue_from:
            package = torch.load(args.continue_from)
            
            self.start_epoch = package['epoch']
            
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(package['state_dict'])
            else:
                self.model.load_state_dict(package['state_dict'])
                
            self.optimizer.load_state_dict(package['optim_dict'])
            
            self.best_loss = package['best_loss']
            self.no_improvement = package['no_improvement']

            self.train_loss[:self.start_epoch] = package['train_loss'][:self.start_epoch]
            self.valid_loss[:self.start_epoch] = package['valid_loss'][:self.start_epoch]
            

    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()
            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec]".format(epoch+1, self.epochs, train_loss, valid_loss, end - start))
            self.train_loss[epoch] = train_loss
            self.valid_loss[epoch] = valid_loss
            
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.no_improvement = 0
                model_path = os.path.join(self.model_dir, "best.pth")
                self.save_model(epoch, model_path)
            else:
                self.no_improvement += 1
                if self.no_improvement >= 5:
                    print("Stop training")
                    break
                if self.no_improvement == 3:
                    optim_dict = self.optimizer.state_dict()
                    lr = optim_dict['param_groups'][0]['lr']
                    
                    print("Learning rate: {} -> {}".format(lr, 0.5 * lr))
                    
                    optim_dict['param_groups'][0]['lr'] = 0.5 * lr
                    self.optimizer.load_state_dict(optim_dict)
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss(train_loss=self.train_loss[:epoch+1], valid_loss=self.valid_loss[:epoch+1], save_path=save_path)
        
    def run_one_epoch(self, epoch):
        self.model.train()
        
        train_loss = 0
        n_train_batch = len(self.train_loader)
    
        for idx, (image, target, pointmap, n_objects) in enumerate(self.train_loader):
            if torch.cuda.is_available():
                image = image.cuda()
                target = {
                    key: target[key].cuda() for key in target.keys()
                }
                pointmap = pointmap.cuda()
                n_objects = n_objects.cuda()

            self.optimizer.zero_grad()
            
            output = self.model(image)
            loss = self.criterion(output, target, pointmap, n_objects)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            
            print("[Epoch {}] iter {}/{} loss: {:.5f}".format(epoch+1, idx+1, n_train_batch, loss.item()), flush=True)
        
        self.model.eval()
        
        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            for idx, (image, target, pointmap, n_objects) in enumerate(self.valid_loader):
                if torch.cuda.is_available():
                    image = image.cuda()
                    target = {
                        key: target[key].cuda() for key in target.keys()
                    }
                    pointmap = pointmap.cuda()
                    n_objects = n_objects.cuda()

                output = self.model(image)
                loss = self.criterion(output, target, pointmap, n_objects, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx == 0:
                    n_image = min(10, len(image))
                    for image_idx in range(n_image):
                        save_dir = os.path.join(self.sample_dir, str(image_idx))
                        os.makedirs(save_dir, exist_ok=True)
                        
                        save_path = os.path.join(save_dir, 'image.png')
                        resized_image = image[image_idx].detach().cpu().numpy().transpose(1,2,0)*255
                        resized_image = resized_image.astype(np.int)
                        resized_image = np.clip(resized_image, 0, 255)
                    
                        plt.figure()
                        plt.imshow(resized_image)
                        plt.savefig(save_path, bbox_inches='tight')
                        plt.close()
                    
                        save_path = os.path.join(save_dir, 'target-heatmap.png')
                        target_heatmap = target['heatmap'][image_idx].detach().cpu().numpy()
                    
                        plt.figure()
                        plt.imshow(target_heatmap)
                        plt.savefig(save_path, bbox_inches='tight')
                        plt.close()
                        
                        save_path = os.path.join(save_dir, 'estimated-heatmap-{}.png'.format(epoch+1))
                        estimated_heatmap = output['heatmap'][image_idx, 0].detach().cpu().numpy()
                    
                        plt.figure()
                        plt.imshow(estimated_heatmap)
                        plt.savefig(save_path, bbox_inches='tight')
                        plt.close()
                        
                        save_path = os.path.join(save_dir, 'estimated-size_x-{}.png'.format(epoch+1))
                        estimated_size_x = output['size'][image_idx, 0].detach().cpu().numpy()
                    
                        plt.figure()
                        plt.imshow(estimated_size_x)
                        plt.savefig(save_path, bbox_inches='tight')
                        plt.close()
        
        train_loss /= n_train_batch
        valid_loss /= n_valid

        return train_loss, valid_loss

    def save_model(self, epoch, model_path='./tmp.pth'):
        package = {}

        package['epoch'] = epoch + 1
        
        if isinstance(self.model, nn.DataParallel):
            package['state_dict'] = self.model.module.state_dict()
        else:
            package['state_dict'] = self.model.state_dict()
            
        package['optim_dict'] = self.optimizer.state_dict()
        
        package['best_loss'] = self.best_loss
        package['no_improvement'] = self.no_improvement
        
        package['train_loss'] = self.train_loss
        package['valid_loss'] = self.valid_loss
        
        torch.save(package, model_path)
        
class Evaluater:
    def __init__(self, model, dataset, args):
        self.model = model
        
        self.dataset = dataset
        
        self._reset(args)
        
    def _reset(self, args):
        self.K = args.K
        self.R = args.R
        self.n_candidate = args.n_candidate
    
        package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(package['state_dict'])
        else:
            self.model.load_state_dict(package['state_dict'])
            
        self.iou = IoU()
        
    def run(self):
        iou_threshold = 0.1
        n_data = len(self.dataset)
        
        self.model.eval()
        
        with torch.no_grad():
        
            for confidence_threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                post_processing = PostProcess(kernel_size=self.K, n_candidate=self.n_candidate, iou_threshold=iou_threshold, confidence_threshold=confidence_threshold)
                total_iou = 0
            
                start = time.time()
                
                for idx, (image, target, bboxes, original_height, original_width) in enumerate(self.dataset):
                    if torch.cuda.is_available():
                        image = image.cuda()
                        target = {
                            key: target[key].cuda() for key in target.keys()
                        }
                    image = image.unsqueeze(dim=0)
                    
                    output = self.model(image)
                    
                    estimated_bboxes = post_processing(output, height=original_height, width=original_width, R=self.R)
                    
                    value = self.iou(estimated_bboxes, bboxes, height=original_width, width=original_width)
                    total_iou += value
                    
                end = time.time()
            
                print("(confidence_threshold = {}, iou_threshold={}) {}, {} [sec]".format(confidence_threshold, iou_threshold, total_iou / n_data, end - start), flush=True)




class PostProcess:
    def __init__(self, kernel_size=1, n_candidate=10, iou_threshold=None, confidence_threshold=None):
        Kh, Kw = _pair(kernel_size)
        
        self.kernel_size = 2*Kh+1, 2*Kw+1
        self.stride = (1, 1)
        
        self.n_candidate = n_candidate
        
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        
    def __call__(self, estimation, height, width, R, iou_threshold=None, confidence_threshold=None):
        """
        Args:
            estimation: {
                'heatmap': (1, 1, H, W) torch.Tensor,
                'size': (1, 2, H, W) torch.Tensor,
            }
        """
        if iou_threshold is None:
            assert self.iou_threshold is not None, "Specify iou_threshold!"
        else:
            assert self.iou_threshold is None, "Specify iou_threshold has already defined!"
            self.iou_threshold = iou_threshold
            
        if confidence_threshold is None:
            assert self.confidence_threshold is not None, "Specify confidence_threshold!"
        else:
            assert self.confidence_threshold is None, "Specify confidence_threshold has already defined!"
            self.confidence_threshold = confidence_threshold
        
        estimated_heatmap = estimation['heatmap']
        estimated_size = estimation['size']
        estimated_local_offset = estimation['local_offset']
        
        batch_size, C, H, W = estimated_heatmap.size()
        
        assert batch_size==1 and C==1, "heatmap.size() must be (1, 1, H, W)"
        
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride
        
        estimated_heatmap = F.pad(estimated_heatmap, (Kw//2,Kw//2,Kh//2,Kh//2))
        
        estimated_local_offset_x, estimated_local_offset_y = estimated_local_offset[:,0], estimated_local_offset[:,1]
        estimated_size_x, estimated_size_y = estimated_size[:,0], estimated_size[:,1]
        
        estimated_heatmap = F.unfold(estimated_heatmap, kernel_size=(Kh, Kw), stride=(Sh, Sw)) # -> (1, 1*Kh*Kw, H'*W'), where H' = (H+Ph-Kh)//Sh+1, W' = (W+Pw-Kw)//Sw+1
        
        estimated_heatmap = estimated_heatmap.squeeze(dim=1).squeeze(dim=0)
        estimated_local_offset_x = estimated_local_offset_x.view(H*W)
        estimated_local_offset_y = estimated_local_offset_y.view(H*W)
        estimated_size_x = estimated_size_x.view(H*W)
        estimated_size_y = estimated_size_y.view(H*W)
        
        heat_max, _ = torch.max(estimated_heatmap, dim=0) # -> (H'*W',)
        mask_max = (heat_max == estimated_heatmap[(Kh*Kw)//2]).float() # -> (H'*W',)
        estimated_heatmap = heat_max * mask_max
        
        _, indices = torch.topk(estimated_heatmap, k=self.n_candidate)
        
        possible_indices = indices.tolist()
        estimated_bboxes = []
        
        while len(possible_indices) > 0:
            target_index = possible_indices.pop(0)
            
            target_bbox = decode(target_index, estimated_heatmap, estimated_local_offset_x, estimated_local_offset_y, estimated_size_x, estimated_size_y, H=H, W=W, R=R, dtype=float)
            
            if target_bbox['confidence'] < self.confidence_threshold:
                break
            
            x1, x2 = target_bbox['box2d']['x1'] * height/(H*R), target_bbox['box2d']['x2'] * height/(H*R)
            y1, y2 = target_bbox['box2d']['y1'] * width/(W*R), target_bbox['box2d']['y2'] * width/(W*R)
            
            if x1 < 0:
                x1 = 0
            if x2 >= width:
                x2 = width - 1
            if y1 < 0:
                y1 = 0
            if y2 >= height:
                y2 = height - 1
            
            target_bbox = {
                'confidence': target_bbox['confidence'],
                'box2d': {
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2)
                }
            }
            
            _possible_indices = []
            
            while len(possible_indices) > 0:
                estimated_index = possible_indices.pop(0)
                estiamted_bbox = decode(estimated_index, estimated_heatmap, estimated_local_offset_x, estimated_local_offset_y, estimated_size_x, estimated_size_y, H=H, W=W, R=R, dtype=float)
                
                x = estimated_index % W
                y = estimated_index // H
                local_offset_x = estimated_local_offset_x[estimated_index].item()
                local_offset_y = estimated_local_offset_y[estimated_index].item()
                size_x = estimated_size_x[estimated_index].item()
                size_y = estimated_size_y[estimated_index].item()
                
                x1, x2 = estiamted_bbox['box2d']['x1'] * height/(H*R), estiamted_bbox['box2d']['x2'] * height/(H*R)
                y1, y2 = estiamted_bbox['box2d']['y1'] * width/(W*R), estiamted_bbox['box2d']['y2'] * width/(W*R)
                
                if x1 < 0:
                    x1 = 0
                if x2 >= width:
                    x2 = width - 1
                if y1 < 0:
                    y1 = 0
                if y2 >= height:
                    y2 = height - 1
                
                estimated_bbox = {
                    'confidence': estimated_heatmap[estimated_index].item(),
                    'box2d': {
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2)
                    }
                }
                
                if IoU.calculate_IoU(estimated_bbox, target_bbox) < self.iou_threshold and estimated_bbox['confidence'] > self.confidence_threshold:
                    _possible_indices.append(estimated_index)
    
            possible_indices = _possible_indices
            estimated_bboxes.append(target_bbox)
        
        return estimated_bboxes
