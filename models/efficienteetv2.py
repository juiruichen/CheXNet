import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetV2, self).__init__()
        self.model = models.efficientnet_v2_l()
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output