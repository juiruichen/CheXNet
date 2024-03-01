import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNet, self).__init__()
        self.model = models.efficientnet_b5()
        
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, out_features=num_classes)
        
    def forward(self, x):
        output = self.model(x)
        return output
        