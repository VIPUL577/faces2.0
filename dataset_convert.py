import torch
import random
import math
import torch.nn as nn
from torchvision.transforms import transforms
from datasets import load_dataset


# device = torch.device("mps")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" )



class FaceDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, widerface_dataset, anchor_generator):
        self.dataset = widerface_dataset
        self.anchor_generator = anchor_generator
        self.all_anchors = self.anchor_generator.generate_anchors()

        self.transform = nn.Sequential(
            transforms.Resize((640, 640)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ).to(device)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        try:
            data= self.dataset[idx]
            image = self.transform((data['image']/255.0).to(torch.float))
            if data['faces']['bbox']!=[] :
                data['faces']['bbox'][:,0]=data['faces']['bbox'][:,0]*(640/1024)
                data['faces']['bbox'][:,1]=data['faces']['bbox'][:,1]*(640/data['image'].shape[1])
                data['faces']['bbox'][:,2]=data['faces']['bbox'][:,2]*(640/1024)
                data['faces']['bbox'][:,3]=data['faces']['bbox'][:,3]*(640/data['image'].shape[1])
                gt_boxes = data['faces']['bbox'].to(device)  # Shape: [num_faces, 4]
            else:
                
                gt_boxes = []
            targets = []
            for level, anchors in enumerate(self.all_anchors):
                level_targets = self.assign_targets(gt_boxes, anchors)
                targets.append(level_targets)
                
            return image.to(device), targets
        except Exception as e:
            print(f"Batch no. {idx}: {e}")
            return self.__getitem__(idx + 1)
    
    def assign_targets(self, gt_boxes, anchors, pos_threshold=0.5, neg_threshold=0.3):
        num_anchors = len(anchors)
        
        if len(gt_boxes) == 0:
            # No faces in image
            return {
                'cls_targets': torch.zeros((num_anchors,1), dtype=torch.long).to(device),
                'cls_weights': torch.zeros(num_anchors).to(device),
                'bbox_targets': torch.zeros(num_anchors, 4).to(device),
                'bbox_weights': torch.zeros(num_anchors).to(device)
            }
        
        ious = self.compute_iou(anchors, gt_boxes)  # [num_anchors, num_gt]
        max_iou_per_anchor, max_iou_indices = ious.max(dim=1)
        cls_targets = torch.zeros(num_anchors, dtype=torch.long).to(device)
        cls_weights = torch.zeros(num_anchors).to(device)
        bbox_targets = torch.zeros(num_anchors, 4).to(device)
        bbox_weights = torch.zeros(num_anchors).to(device)
        positive_mask = max_iou_per_anchor > pos_threshold
        cls_targets[positive_mask] = 1  # Face class
        bbox_weights[positive_mask] = 1.0
        cls_weights[positive_mask] = 1.0
        
        negative_mask = max_iou_per_anchor < neg_threshold
        cls_targets[negative_mask] = 0
        cls_weights[negative_mask] = 1.0
        if positive_mask.sum() > 0:
            positive_anchors = anchors[positive_mask]
            assigned_gt = gt_boxes[max_iou_indices[positive_mask]]
            bbox_targets[positive_mask] = self.encode_bbox_targets(assigned_gt, positive_anchors)
        return {
            'cls_targets': cls_targets.reshape((cls_targets.shape[0],1)),
            'cls_weights': cls_weights,
            'bbox_targets': bbox_targets,
            'bbox_weights': bbox_weights
        }
    
    def compute_iou(self, anchors, gt_boxes):
        # anchors: [num_anchors, 4] in [x1, y1, w, b] format
        # gt_boxes: [num_gt, 4] in [x1, y1, w, b] format
        
        num_anchors = anchors.size(0)
        num_gt = gt_boxes.size(0)
        # print(num_anchors.size)
        # print(num_gt)
        
        anchors = anchors.unsqueeze(1).expand(num_anchors, num_gt, 4)
        gt_boxes = gt_boxes.unsqueeze(0).expand(num_anchors, num_gt, 4)
        
        anchor_area = anchors[:, :, 2] * anchors[:, :, 3]
        gt_area     = gt_boxes[:, :, 2] * gt_boxes[:, :, 3]

        inter_x1 = torch.max(anchors[:, :, 0], gt_boxes[:, :, 0])   #x1   
        inter_y1 = torch.max(anchors[:, :, 1], gt_boxes[:, :, 1])   #y1
        inter_x2 = torch.min(anchors[:, :, 2]+anchors[:,:,0], gt_boxes[:, :, 2]+gt_boxes[:,:,0])   #x2
        inter_y2 = torch.min(anchors[:, :, 3]+anchors[:,:,1], gt_boxes[:, :, 3]+gt_boxes[:,:,1])   #y2
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area  = anchor_area + gt_area - inter_area

        iou = inter_area / torch.clamp(union_area, min=1e-6)
  
        return iou.to(device)
    
    def encode_bbox_targets(self, gt_boxes, anchors):
        # Convert to center format
        anchor_widths = anchors[:, 2]
        anchor_heights = anchors[:, 3]
        anchor_cx = anchors[:, 0] + 0.5 * anchor_widths
        anchor_cy = anchors[:, 1] + 0.5 * anchor_heights
        
        gt_widths = gt_boxes[:, 2]
        gt_heights = gt_boxes[:, 3]
        gt_cx = gt_boxes[:, 0] + 0.5 * gt_widths
        gt_cy = gt_boxes[:, 1] + 0.5 * gt_heights
        
        # Encode as offsets
        target_dx = (gt_cx - anchor_cx) / anchor_widths
        target_dy = (gt_cy - anchor_cy) / anchor_heights
        target_dw = torch.log(gt_widths / anchor_widths)
        target_dh = torch.log(gt_heights / anchor_heights)
        x=torch.stack([target_dx, target_dy, target_dw, target_dh], dim=1).to(device)
        return x
    
class AnchorGenerator:
    def __init__(self):
        # Define scales and aspect ratios for each FPN level
        self.scales = [128, 64, 32, 16]
        self.aspect_ratios = [0.5, 1.0, 2.0]  # Common face aspect ratios
        self.anchor_scales = [2**0, 2**(1/3), 2**(2/3), 0.5]  # Sub-octave scales
        
        # FPN level strides
        self.strides = [32, 16, 8, 4 ]  # Corresponding to your FPN levels
        
    def generate_anchors(self, feature_map_sizes = [(20,20),(40,40),(80,80),(160,160)]):
        all_anchors = []
        # feature_map_sizes = [(20,20),(40,40),(80,80),(160,160)] example
        for level, (h, w) in enumerate(feature_map_sizes):
            level_anchors = self.generate_level_anchors(
                h, w, self.scales[level], self.strides[level]
            )
            all_anchors.append(level_anchors)
            
        return all_anchors
    
    def generate_level_anchors(self, h, w, base_size, stride):
        anchors = []
        
        for i in range(h):
            for j in range(w):
                cx = (j + 0.5) * stride
                cy = (i + 0.5) * stride
                
                for aspect_ratio in self.aspect_ratios:
                    for scale in self.anchor_scales:
                        anchor_w = base_size * scale * math.sqrt(aspect_ratio)
                        anchor_h = base_size * scale / math.sqrt(aspect_ratio)

                        x1 = cx - anchor_w / 2
                        y1 = cy - anchor_h / 2
                        anchors.append([x1, y1,anchor_w ,anchor_h ])
        
        return torch.tensor(anchors, dtype=torch.float32,device=device)
    


