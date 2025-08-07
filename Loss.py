import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lossfunction(nn.Module):
    def __init__(self , alpha= 0.25 , gamma= 2, lambd = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.BCEloss = nn.BCEWithLogitsLoss(reduction='none').to(device)
        self.sigmoid = nn.Sigmoid().to(device)

    def classloss(self, p, cls_targets, weights):
        # p = p.clamp(min=1e-8, max = 1-(1e-8))
        pa = self.sigmoid(p)
        alpha = (self.alpha* cls_targets) + ((1-self.alpha)*(1- cls_targets))
        pt = ((1-pa) * (1-cls_targets)) + (pa * cls_targets)
    
        p = self.BCEloss(p,cls_targets.to(torch.float))
        # print(f"ff{p.shape}")
        loss = (alpha*((1-pt)**self.gamma)*p)*weights.unsqueeze(-1)
        pos = (cls_targets.sum()).clamp(1.0)
        # print(loss.shape)
        # print(f"cls--raw:{(pos_loss+neg_loss).sum()}\n p:{pos}")
        return loss.sum()/pos
    
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
            if len(targeta["cls_targets"].shape)==2:
                targeta["cls_targets"] = targeta["cls_targets"].unsqueeze(0)
            if targeta['cls_weights'].sum() > 0:
                cls_loss += self.classloss(predicta["cls"], targeta["cls_targets"].to(device),targeta["cls_weights"].to(device))
            if targeta['bbox_weights'].sum() > 0:
                bbox_loss += self.reg_loss(predicta["bbox"],targeta["bbox_targets"].to(device),targeta['bbox_weights'].to(device))
                # k+=targeta['bbox_weights'].sum()
            # print(f"k:{k}")
        # print((bbox_loss/cls_loss)*1.0)
        # print(f'''box loss:{bbox_loss}\ncls loss:{cls_loss}''')
        return bbox_loss + (self.lambd*cls_loss)
            
            