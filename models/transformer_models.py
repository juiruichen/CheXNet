import timm
import torch
import torch.nn as nn

''' ViT '''
class ViT(nn.Module):
    
    def __init__(self,):
        super(ViT, self).__init__()
        self.model = timm.create_model('vit_base_patch8_224', pretrained=True, img_size=128, in_chans=4, num_classes=2,)
        
    def forward(self, x):
        return self.model(x)

''' Swin Transformer ''' 
class Swin(nn.Module):
    
    def __init__(self,):
        super(Swin, self).__init__()
        self.model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, img_size=128, in_chans=4, num_classes=2)
    
    def forward(self, x):
        return self.model(x)

''' Swin Transformer V2 '''
class SwinV2(nn.Module):
    
    def __init__(self,):
        super(SwinV2, self).__init__()
        self.model = timm.create_model("swinv2_base_window8_256", pretrained=True, img_size=128, in_chans=4, num_classes=2)
    
    def forward(self, x):
        return self.model(x)
        
''' DeiT '''
class DeiT(nn.Module):
    
    def __init__(self,):
        super(DeiT, self).__init__()
        self.model = timm.create_model("deit_base_patch16_224", pretrained=True, img_size=128, in_chans=4, num_classes=2)
        
    def forward(self, x):
        return self.model(x)