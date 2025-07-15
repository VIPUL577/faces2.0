import torch
import torch.nn as nn
import numpy as np

class Lossfunction(nn.Module):
    def __init__(self , alpha= 0.25 , gamma= 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
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
            targeta = targets[3-level]
            cls_loss += self.classloss(predicta["cls"], targeta["cls_targets"])
            bbox_loss += self.reg_loss(predicta["bbox"],targeta["bbox_targets"]) if targeta["bbox_weights"].sum() > 0 else 0 
            
        return cls_loss + bbox_loss      
            
            