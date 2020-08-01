import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class IoU:
    def __init__(self):
        pass
        
    def __call__(self, estimated_bboxes, target_bboxes, height, width):
        """
        Args:
            <estiamted, target>_bboxes: [
                {
                    'confidence': confidence,
                    'box2d': {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    }
                },
                ...
            ]
        """
        n_estimated_bbox = len(estimated_bboxes)
        n_target_bbox = len(target_bboxes)
        
        if n_estimated_bbox == 0:
            n_estimated_bbox = 1
        if n_target_bbox == 0:
            n_target_bbox = 1
            
        estimated_bbox_field = np.zeros((n_estimated_bbox, height, width), dtype=np.int)
        target_bbox_field = np.zeros((n_target_bbox, height, width), dtype=np.int)
        
        for idx, estimated_bbox in enumerate(estimated_bboxes):
            bbox_field = fill_bbox(estimated_bbox, height, width)
            estimated_bbox_field[idx] = bbox_field
            
        for idx, target_bbox in enumerate(target_bboxes):
            bbox_field = fill_bbox(target_bbox, height, width)
            target_bbox_field[idx] = bbox_field
            
        estimated_bbox_field = np.max(estimated_bbox_field, axis=0)
        target_bbox_field = np.max(target_bbox_field, axis=0)
        
        intersection = estimated_bbox_field & target_bbox_field
        union = estimated_bbox_field | target_bbox_field
        
        intersection = np.sum(intersection)
        union = np.sum(union)
        
        if union == 0:
            return 0

        return  intersection / union
        
        
    @staticmethod
    def calculate_IoU(estimated_bbox, target_bbox):
        """
        Args:
            bbox: {
                'box2d': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                }
            }
        """
        height = max(estimated_bbox['box2d']['y2'], target_bbox['box2d']['y2']) + 1
        width = max(estimated_bbox['box2d']['x2'], target_bbox['box2d']['x2']) + 1

        estimated_field = fill_bbox(estimated_bbox, height, width)
        target_field = fill_bbox(target_bbox, height, width)
        
        intersection = estimated_field & target_field
        union = estimated_field | target_field
        
        intersection = np.sum(intersection)
        union = np.sum(union)
        
        if union == 0:
            return 0

        return intersection / union

        
    @classmethod
    def estimate_bboxes(cls, estimation, R, kernel_size=1, n_candidate=10, confidence_threshold=0.9, iou_threshold=0.1):
        """
        Args:
            estimation: {
                'heatmap': (1, 1, H, W) torch.Tensor,
                'size': (1, 2, H, W) torch.Tensor,
            }
        """
        estimated_heatmap = estimation['heatmap']
        estimated_size = estimation['size']
        estimated_local_offset = estimation['local_offset']
        
        batch_size, C, H, W = estimated_heatmap.size()
        
        assert batch_size==1 and C==1, "heatmap.size() must be (1, 1, H, W)"
        
        Kh, Kw = _pair(kernel_size)
        Kh, Kw = 2*Kh+1, 2*Kw+1
        Sh, Sw = 1, 1
        
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
        
        _, indices = torch.topk(estimated_heatmap, k=n_candidate)
        
        possible_indices = indices.tolist()
        estimated_bboxes = []
        
        while len(possible_indices) > 0:
            target_index = possible_indices.pop(0)
            target_bbox = decode(target_index, estimated_heatmap, estimated_local_offset_x, estimated_local_offset_y, estimated_size_x, estimated_size_y, H=H, W=W, R=R, dtype=int)
            
            if target_bbox['confidence'] < confidence_threshold:
                break
            
            _possible_indices = []
            
            while len(possible_indices) > 0:
                estimated_index = possible_indices.pop(0)
                estimated_bbox = decode(estimated_index, estimated_heatmap, estimated_local_offset_x, estimated_local_offset_y, estimated_size_x, estimated_size_y, H=H, W=W, R=R, dtype=int)
                
                if cls.calculate_IoU(estimated_bbox, target_bbox) < iou_threshold and estimated_bbox['confidence'] > confidence_threshold:
                    _possible_indices.append(estimated_index)
    
            possible_indices = _possible_indices
            estimated_bboxes.append(target_bbox)
        
        return estimated_bboxes


def fill_bbox(bbox, height, width):
    """
    Args:
        bbox: {
            'confidence': confidence,
            'box2d': {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            }
        }
    """
    bbox_field = np.zeros((height, width), dtype=np.int)
    
    x1, x2 = bbox['box2d']['x1'], bbox['box2d']['x2']
    y1, y2 = bbox['box2d']['y1'], bbox['box2d']['y2']

    bbox_field[y1:y2+1, x1:x2+1] = 1

    return bbox_field

    
def decode(index, heatmap, local_offset_x, local_offset_y, size_x, size_y, H, W, R, dtype=int):
    confidence = heatmap[index].item()
    
    x = index % W
    y = index // H
    local_offset_x = local_offset_x[index].item()
    local_offset_y = local_offset_y[index].item()
    size_x = size_x[index].item()
    size_y = size_y[index].item()
    
    x, y = x + local_offset_x, y + local_offset_y
    x, y = x * R, y * R
    
    x1, x2 = x - size_x/2, x + size_x/2
    y1, y2 = y - size_y/2, y + size_y/2
    
    bbox = {
        'confidence': confidence,
        'box2d': {
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2)
        }
    }
    
    return bbox
    
    
if __name__ == '__main__':
    iou = IoU()
    
    
    estimated_bboxes = [{
        'box2d': {
            'x1': 1,
            'y1': 51,
            'x2': 100,
            'y2': 100
        }
    
    }]
    target_bboxes = [{
        'box2d': {
            'x1': 51,
            'y1': 1,
            'x2': 150,
            'y2': 100
        }
    }, {
        'box2d': {
            'x1': 101,
            'y1': 51,
            'x2': 200,
            'y2': 150
        }
    }]
    
    result = iou(estimated_bboxes, target_bboxes, height=400, width=400)
    print(result)

    
