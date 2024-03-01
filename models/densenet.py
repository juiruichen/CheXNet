import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet201(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DenseNet201, self).__init__()

        self.model = models.densenet201()
        self.model.classifier = nn.Linear(1920, num_classes)
        
    def forward(self, x):
        output = self.model(x)
        return output