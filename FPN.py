import torch
import torch.nn as nn
import numpy as np
from mobilenetv2 import ConvBN

# device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Features(nn.Module):
    def __init__(self, model, layers:list):
        super().__init__()
        self.model = model
        self.layers = layers
        self.feature = {}
        self.hooks = []
        for layer in layers:
            lay = dict(model.named_modules())[layer]
            hook = lay.register_forward_hook(self.get_features(layer))
            self.hooks.append(hook)
    def get_features(self,layer_name):
        def hook (model, input, output):
            self.feature[layer_name] = output
        return hook
    def extract(self,x):
        _ = self.model(x)
        return self.feature
    def remove(self):
        for hook in self.hooks:
            hook.remove()
            
class FPNetwork(nn.Module):
    def __init__(self, 
                 in_channels:dict = {'3':24,
                                     '6':32,
                                     '13':96,
                                     '18':1280}, 
                 out_channels:int = 256):
        super().__init__()
        # features should be in ascending order: ex [3,6,13,18] 3 stands for 3rd layer etc
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest').to(device)
        self.outchannels = out_channels
        self.convs = nn.ModuleDict()
        self.final_conv = nn.ModuleDict()
        for level,key in enumerate(list(in_channels.keys())):
            self.convs.add_module(str(key).replace('.','_'),ConvBN(in_channels=in_channels[key],out_channels=self.outchannels, kernel_size= 1,stride=1,activation=False).to(device))
            self.final_conv.add_module(str(key).replace('.','_'),ConvBN(in_channels=out_channels,out_channels=out_channels, kernel_size= 3,stride=1,padding=1,activation=False).to(device))
            
    def forward(self, features:dict):
        self.keys = list(features.keys())
        self.features = features
        self.output = {}
        
        for i in range (len(self.keys)-1,-1,-1):
            feature = self.features[self.keys[i]]
            if i == len(self.keys)-1:
                # print(self.keys[i].replace('.','_'))
                self.output[self.keys[i]] = self.final_conv[self.keys[i].replace('.','_')](self.convs[self.keys[i].replace('.','_')](feature))
                continue
            x = self.convs[self.keys[i].replace('.','_')](feature)
            self.output[self.keys[i]] = self.final_conv[self.keys[i].replace('.','_')](x + self.upsample(self.output[self.keys[i+1]]))
            
        return self.output
        

class classificationhead(nn.Module):
    def __init__(self, channels, num_anchors, num_of_classes):
        super().__init__()
        self.channels = channels
        self.anchors = num_anchors
        self.num_of_classes = num_of_classes
        # self.sigmoid = nn.Sigmoid().to(device)

        self.model = nn.Sequential(
            ConvBN(in_channels=channels,out_channels=channels, kernel_size= 3,stride=1,padding=1).to(device),
            ConvBN(in_channels=channels,out_channels=channels, kernel_size= 3,stride=1,padding=1).to(device),
            ConvBN(in_channels=channels,out_channels=channels, kernel_size= 3,stride=1,padding=1).to(device),
            ConvBN(in_channels=channels,out_channels=channels, kernel_size= 3,stride=1,padding=1).to(device),
            ConvBN(in_channels= self.channels, out_channels= self.anchors*self.num_of_classes, kernel_size= 3,stride=1,padding=1,pi=0.01,activation=False)
        ).to(device)
        
    def forward(self,x):
        x = self.model(x)
        _x = x.view(x.shape[0],self.anchors,self.num_of_classes, x.shape[2], x.shape[3])
        f_x = _x.permute(0, 1, 3, 4, 2).contiguous()
        x = (f_x.reshape((x.shape[0],f_x.shape[1]*f_x.shape[2]*f_x.shape[3],self.num_of_classes))) 
        return x
    
class bboxhead(nn.Module):
    def __init__(self, channels, num_anchors):
        super().__init__()
        self.channels = channels
        self.anchors = num_anchors
        self.linear = nn.Conv2d(in_channels=self.channels,out_channels= self.anchors*4, kernel_size= 3,stride=1,padding=1 )
        self.model = nn.Sequential(
            ConvBN(in_channels=channels,out_channels=channels, kernel_size= 3,stride=1,padding=1).to(device),
            ConvBN(in_channels=channels,out_channels=channels, kernel_size= 3,stride=1,padding=1).to(device),
            ConvBN(in_channels=channels,out_channels=channels, kernel_size= 3,stride=1,padding=1).to(device),
            ConvBN(in_channels=channels,out_channels=channels, kernel_size= 3,stride=1,padding=1).to(device),
            self.linear
        ).to(device)
        
    def forward(self,x):
        x = self.model(x)
        _x = x.view(x.shape[0], self.anchors, 4, x.shape[2], x.shape[3])
        f_x = _x.permute(0, 1, 3, 4, 2).contiguous()
        x= f_x.reshape((x.shape[0],f_x.shape[1]*f_x.shape[2]*f_x.shape[3],4))
        return x