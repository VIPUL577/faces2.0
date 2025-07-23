import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init

# device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBN(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size, stride, bais=True,padding=0,pi = 0.5,activation = True):
        super().__init__()
        layer = []
        layer.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bais,stride=stride,padding=padding))
        layer.append(nn.BatchNorm2d(num_features=out_channels))
        if activation:
            layer.append(nn.ReLU(inplace=True))
        init.xavier_uniform_(layer[0].weight) 
        pi= torch.tensor(pi)
        init.constant_(layer[0].bias, -torch.log((1-pi)/pi))
        self.model = nn.Sequential(*layer)#.to(device)
    def forward(self,x):
        return self.model(x)
    
class Inverted_Residual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=2 ):
        super().__init__()
        if stride != 1 and stride != 2:
            raise ValueError("Stride should be 1 or 2")
        
        self.inchannels = in_channels
        self.outchannels = out_channels
        
        self.convop = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                      kernel_size, stride, padding=1,
                      groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, out_channels, 1),
            nn.BatchNorm2d(out_channels)).to(device)

        self.is_residual = True if stride == 1 else False
    def forward(self, x):
        if self.is_residual:
            _x = nn.Conv2d(self.inchannels,self.outchannels,1).to(device)(x)
            return _x + self.convop(x)
        else:
            return self.convop(x)
        
class MobileNetv2(nn.Module):
                        #  t, c, n, s
    structure = np.array([[1,16 ,1,1],
                          [6,24 ,2,2],
                          [6,32 ,3,2],
                          [6,64 ,4,2],
                          [6,96 ,3,1],
                          [6,160,3,2],
                          [6,320,1,1]
                          ]) 
    def __init__(self):
        super().__init__()
        layer=[]
        st = self.structure
        layer.append(ConvBN(3,32,3,2))
        for i in range (0,7):
            inchannels = st[i-1,1] if (i!=0) else 32
            layer.append(
                Inverted_Residual(in_channels=inchannels,
                                  out_channels=st[i,1],
                                  expansion_factor=st[i,0],
                                  stride=st[i,3])
            )
            for j in range (1,st[i,2]):
                layer.append(
                Inverted_Residual(in_channels=st[i,1],
                                  out_channels=st[i,1],
                                  expansion_factor=st[i,0],
                                  stride=1)
                )
        layer.append(ConvBN(320,1280,1,1))
        self.model = nn.Sequential(*layer).to(device)
    def forward(self,x):
        return self.model(x)
    