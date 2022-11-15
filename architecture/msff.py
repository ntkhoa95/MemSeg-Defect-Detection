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
    """
    Implementation of `MS-FFN <https://arxiv.org/pdf/2205.00908.pdf>
    Multi-scale Feature Fusion Module
    """
    def __init__(self):
        super(MSFF, self).__init__()
        self.block3 = MSFFBlock(128)
        self.block2 = MSFFBlock(256)
        self.block1 = MSFFBlock(512)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.upconv32 = nn.Sequential(
            nn.Upsample(scale_factor=1/2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        )
        self.upconv21 = nn.Sequential(
            nn.Upsample(scale_factor=1/2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, features):
        """
        Arguments:
            features  (List)  : [ci3, ci2, ci1]
            ci1       (Tensor): an [N, 512, 16, 16]
            ci2       (Tensor): an [N, 256, 32, 32]
            ci3       (Tensor): an [N, 128, 64, 64]
        """
        ci3, ci2, ci1 = features
        
        # MSFF Module
        ci1_k = self.block1(ci1)      # [N, 256, 16, 16]
        ci2_k = self.block2(ci2)      # [N, 128, 32, 32]
        ci3_k = self.block3(ci3)      # [N, 64, 64, 64]

        # ci2_f = ci2_k + self.upconv32(ci1_k)       # [N, 128, 32, 32]
        # ci3_f = ci3_k + self.upconv21(ci2_f)       # [N,  64, 64, 64]

        ci2_sigma = ci2_k + self.upconv32(ci3_k)
        ci1_sigma = ci1_k + self.upconv21(ci2_sigma)

        # spatial attention
        # mask 
        # m1 = ci1[:,256:,...].mean(dim=1, keepdim=True)
        # m2 = ci2[:,128:,...].mean(dim=1, keepdim=True) * self.upsample(m1)
        # m3 = ci3[:, 64:,...].mean(dim=1, keepdim=True) * self.upsample(m2)

        m1 = ci1[:, :, ...].mean(dim=1, keepdim=True)
        m2 = ci2[:, :, ...].mean(dim=1, keepdim=True) * self.upsample(m1)
        m3 = ci3[:, :, ...].mean(dim=1, keepdim=True) * self.upsample(m2)

        ci1_out = ci1_sigma * m1
        ci2_out = ci2_sigma * m2
        ci3_out = ci3_k * m3
        
        return [ci3_out, ci2_out, ci1_out]

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # build feature extractor
    feature_extractor = create_model(
        'resnet18', 
        pretrained    = True, 
        features_only = True)
    ## freeze weight of layer1,2,3
    for l in ['layer1','layer2','layer3']:
        for p in feature_extractor[l].parameters():
            p.requires_grad = False

    batch_size = 4
    # Inputs [torch.Size([8, 128, 64, 64]) torch.Size([8, 256, 32, 32]) torch.Size([8, 512, 16, 16])]
    inputs_0 = torch.randn(batch_size, 128, 64, 64)
    inputs_1 = torch.randn(batch_size, 256, 32, 32)
    inputs_2 = torch.randn(batch_size, 512, 16, 16)
    inputs = [inputs_0, inputs_1, inputs_2]
    model = MSFF()
    model(inputs)
