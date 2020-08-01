import os
import numpy as np
import cv2
import torch
import torch.utils.data

import matplotlib.pyplot as plt

EPS = 1e-9

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, annotation_path, H=256, W=256, R=4, G=5):
        self.H, self.W = H, W
        self.R = R
        self.G = G
        
        annotations = []
        
        with open(annotation_path, 'r') as f:
            line = f.readline()
            n_data = int(line.strip())
            
            line = f.readline()
            
            for n in range(n_data):
                annotation = {}
                
                line = f.readline()
                relative_path, x1, y1, w, h = line.split()
                
                image_path = os.path.join(image_root, relative_path)
                
                annotation['path'] = image_path
                annotation['n_objects'] = 1
                annotation['object'] = []

                x1, y1 = int(x1), int(y1)
                w, h = int(w), int(h)
                
                x2, y2 = x1 + w, y1 + h
                bbox = {
                    'box2d': {
                        'x1': x1,
                        'x2': x2,
                        'y1': y1,
                        'y2': y2
                    }
                }
                annotation['object'].append(bbox)

                annotations.append(annotation)
                
        self.annotations = annotations
            
        
    def __getitem__(self, idx):
        H, W = self.H, self.W
        R = self.R
        G = self.G
    
        annotation = self.annotations[idx]
        
        image = cv2.imread(annotation['path'])
        
        height, width, _ = image.shape
        
        annotation['height'], annotation['width'] = height, width
        
        image = cv2.resize(image, (W, H))
        image = image[..., ::-1] # BGR to RGB
        
        # Coordinates
        x = np.arange(0, W//R, dtype=np.float)
        y = np.arange(0, H//R, dtype=np.float)
        x, y = np.meshgrid(x, y)
        
        n_objects = annotation['n_objects']
        bboxes = annotation['object']
        
        target_heatmaps = None
        target_local_offsets = None
        target_sizes = None
        pointmap = np.zeros((H//R, W//R), np.float)

        for bbox in bboxes:
            x1 = bbox['box2d']['x1']
            y1 = bbox['box2d']['y1']
            x2 = bbox['box2d']['x2']
            y2 = bbox['box2d']['y2']
            y1, x1 = y1*H/height, x1*W/width
            y2, x2 = y2*H/height, x2*W/width

            # Heatmap
            center_x, center_y = int((x1+x2)/2)//R, int((y1+y2)/2)//R
            
            if center_x < 0:
                center_x = 0
            elif center_x >= W//R:
                center_x = W//R - 1
            if center_y < 0:
                center_y = 0
            elif center_y >= H//R:
                center_y = H//R - 1
            
            size_x, size_y = x2-x1, y2-y1
            dif_x, dif_y = x-float(center_x), y-float(center_y)
            var_x, var_y = size_x/(G*R), size_y/(G*R)
            target_heatmap = np.exp(-(var_y*dif_x**2+var_x*dif_y**2)/(2 * var_x * var_y + EPS))
            
            # Local offset
            target_local_offset = np.zeros((2, H//R, W//R), dtype=np.float)
            target_local_offset[0][center_y, center_x] = ((x1+x2)/2)/R - center_x
            target_local_offset[1][center_y, center_x] = ((y1+y2)/2)/R - center_y

            # Size
            target_size = np.zeros((2, H//R, W//R), dtype=np.float)
            target_size[0][center_y, center_x] = size_x
            target_size[1][center_y, center_x] = size_y

            # Pointmap
            pointmap[center_y, center_x] = 1

            if target_heatmaps is None:
                target_heatmaps = target_heatmap[None]
                target_local_offsets = target_local_offset[None]
                target_sizes = target_size[None]
            else:
                target_heatmaps = np.concatenate([target_heatmaps, target_heatmap[None]], axis=0)
                target_local_offsets = np.concatenate([target_local_offsets, target_local_offset[None]], axis=0)
                target_sizes = np.concatenate([target_sizes, target_size[None]], axis=0)

        if target_heatmaps is None:
            target_heatmap = np.zeros((H//R, W//R), dtype=np.float)
            target_local_offset = np.zeros((2, H//R, W//R), dtype=np.float)
            target_size = np.zeros((2, H//R, W//R), dtype=np.float)
        else:
            target_heatmap = np.max(target_heatmaps, axis=0)
            target_local_offset = np.max(target_local_offsets, axis=0)
            target_size = np.max(target_sizes, axis=0)
        
        image = image.transpose(2,0,1) / 255
        image = torch.Tensor(image.copy())
        target = {
            'heatmap': torch.Tensor(target_heatmap),
            'local_offset': torch.Tensor(target_local_offset),
            'size': torch.Tensor(target_size)
        }
        pointmap = torch.Tensor(pointmap)
        
        return image, target, pointmap, n_objects

    def __len__(self):
        return len(self.annotations)
        

class EvalDataset(TrainDataset):
    def __init__(self, image_root, annotation_path, H=256, W=256, R=4, G=5):
        super().__init__(image_root, annotation_path, H=H, W=W, R=R, G=G)
        
    def __getitem__(self, idx):
        image, target, pointmap, n_objects = super().__getitem__(idx)
        
        annotation = self.annotations[idx]

        bboxes = annotation['object']
        
        height, width = annotation['height'], annotation['width']
        
        return image, target, bboxes, height, width

class TrainDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EvalDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
