import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Lossfunction(nn.Module):
    def __init__(self , alpha= 0.25 , gamma= 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def classloss(self, cls_pred, cls_targets):
        """Focal loss for classification"""
        # ce_loss = F.cross_entropy(cls_pred, cls_targets.to(torch.float), reduction='none')
        # pt = torch.exp(-ce_loss)
        # focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        # return focal_loss.mean()
    def classloss(self, cls_pred, cls_targets):
        loss = -self.alpha*((cls_targets*((1-cls_pred)**self.gamma)*torch.log(cls_pred))+((1-cls_targets)*((cls_pred)**self.gamma)*torch.log(1-cls_pred)))
        return loss.mean()
    
    def reg_loss(self , bbox_pred, bbox_targets, weights):
        diff = bbox_pred - bbox_targets
        loss = torch.where(torch.abs(diff)<1, 0.5*diff*diff,torch.abs(diff) - 0.5 )
        return (loss*weights.unsqueeze(-1)).mean()
    
    def forward(self, predicted:dict , targets:list):
        cls_loss = 0
        bbox_loss = 0 
        for level,key in enumerate(predicted.keys()):
            predicta = predicted[key]
            targeta = targets[level]
            # print(f"Level {level}, Key: {key}")
            # print(f"Pred cls shape: {predicta['cls'].shape}")
            # print(f"Target cls shape: {targeta['cls_targets'].shape}")
            # print(f"Pred bbox shape: {predicta['bbox'].shape}")
            # print(f"Target bbox shape: {targeta['bbox_targets'].shape}")
            # print(f"Bbox weights shape: {targeta['bbox_weights'].shape}")
            cls_loss += self.classloss(predicta["cls"], targeta["cls_targets"].to(device))
            if targeta['bbox_weights'].sum() > 0:
                bbox_loss += self.reg_loss(predicta["bbox"],targeta["bbox_targets"].to(device),targeta['bbox_weights'].to(device))
            
        return cls_loss + bbox_loss      
            
            