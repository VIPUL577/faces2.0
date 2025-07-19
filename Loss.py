import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Lossfunction(nn.Module):
    def __init__(self , alpha= 0.1 , gamma= 2, lambd = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
    def classloss(self, p, cls_targets):
        
        pos_loss = -self.alpha * ((1-p)**self.gamma) * cls_targets       * torch.log(p)
        neg_loss = -(1-self.alpha) * (p**self.gamma) * (1-cls_targets)   * torch.log(1-p)
        pos = (cls_targets.sum()).clamp(1.0)
        # print(f"cls--raw:{(pos_loss+neg_loss).sum()}\n p:{pos}")
        return (pos_loss+neg_loss).sum()/pos
    
    def reg_loss(self , bbox_pred, bbox_targets, weights):
        diff = bbox_pred - bbox_targets
        loss = (torch.where(torch.abs(diff)<1, 0.5*diff*diff,torch.abs(diff) - 0.5 )*weights.unsqueeze(-1)).sum()
        pos = weights.sum().clamp(min=1.)  
        return (loss/pos)
    
    def forward(self, predicted:dict , targets:list):
        cls_loss = 0
        bbox_loss = 0 
        
        for level,key in enumerate(predicted.keys()):
            predicta = predicted[key]
            targeta = targets[level]
            cls_loss += self.classloss(predicta["cls"], targeta["cls_targets"].to(device))
            if targeta['bbox_weights'].sum() > 0:
                bbox_loss += self.reg_loss(predicta["bbox"],targeta["bbox_targets"].to(device),targeta['bbox_weights'].to(device))
                # k+=targeta['bbox_weights'].sum()
            # print(f"k:{k}")
        # print((bbox_loss/cls_loss)*1.0)
        print(f'''box loss:{bbox_loss}\ncls loss:{cls_loss}''')
        return cls_loss + (self.lambd*bbox_loss)
            
            