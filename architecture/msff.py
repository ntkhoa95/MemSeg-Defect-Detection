import torch, math
import torch.nn as nn
import torch.nn.functional as F
from .coordatt import CoordAtt
from timm import create_model

class MSFFBlock(nn.Module):
    def __init__(self, in_channel):
        super(MSFFBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.attn = CoordAtt(in_channel, in_channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x_conv = self.conv1(x)
        x_att = self.attn(x)
        
        x = x_conv * x_att
        x = self.conv2(x)
        return x

    
class MSFF(nn.Module):
    def __init__(self):
        super(MSFF, self).__init__()
        self.blk1 = MSFFBlock(128)
        self.blk2 = MSFFBlock(256)
        self.blk3 = MSFFBlock(512)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.upconv32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        )
        self.upconv21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, features):
        # features = [level1, level2, level3]
        f1, f2, f3 = features 
        
        # MSFF Module
        f1_k = self.blk1(f1)
        f2_k = self.blk2(f2)
        f3_k = self.blk3(f3)

        f2_f = f2_k + self.upconv32(f3_k)
        f1_f = f1_k + self.upconv21(f2_f)

        # spatial attention
        
        # mask 
        m3 = f3[:,256:,...].mean(dim=1, keepdim=True)
        m2 = f2[:,128:,...].mean(dim=1, keepdim=True) * self.upsample(m3)
        m1 = f1[:,64:,...].mean(dim=1, keepdim=True) * self.upsample(m2)
        
        f1_out = f1_f * m1
        f2_out = f2_f * m2
        f3_out = f3_k * m3
        
        return [f1_out, f2_out, f3_out]



if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # build feature extractor
    feature_extractor = create_model(
        'resnet18', 
        pretrained    = True, 
        features_only = True).to(device)
    ## freeze weight of layer1,2,3
    for l in ['layer1','layer2','layer3']:
        for p in feature_extractor[l].parameters():
            p.requires_grad = False

    batch_size = 4
    inputs = torch.randn(batch_size, 3, 256, 256)
    features = feature_extractor
    model = MSFF()
    model.to(device)
    model(inputs)
