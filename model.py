import torch
import torch.nn as nn
import numpy as np
from mobilenetv2 import MobileNetv2
from FPN import Features, FPNetwork , classificationhead , bboxhead

# device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model = model.features.to(device)

p=torch.rand(8,3,640,640,device=device)

extractor = Features(model,['3','6', '13','18'])
features = extractor.extract(p)
topdown = FPNetwork(out_channels=256)
newfeatures = topdown(features)
classifier = classificationhead(channels=256, num_anchors= 12, num_of_classes= 1)
bboxregression = bboxhead(channels= 256 , num_anchors= 12)
output = {}
for key in list(newfeatures.keys()):
    temp = {}
    temp["bbox"] = bboxregression(newfeatures[key])
    temp["cls"] = classifier(newfeatures[key])
    output[key] = temp


# def forward(p):
#     features = extractor.extract(p)
#     newfeatures = topdown(features)
#     output = {}
#     for key in list(newfeatures.keys()):
#         temp = {}
#         temp["bbox"] = bboxregression(newfeatures[key])
#         temp["cls"] = classifier(newfeatures[key])
#         output[key] = temp
#     return output