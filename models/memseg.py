import torch
import torch.nn as nn
import torch.nn.functional as F
from .msff import MSFF
from .decoder import Decoder



class MemSeg(nn.Module):
    def __init__(self, memory_module, encoder):
        super(MemSeg, self).__init__()
        self.encoder = encoder
        self.memory_module = memory_module
        self.msff = MSFF()
        self.decoder = Decoder()

    def forward(self, inputs):
        features = self.encoder(inputs)
        f_in  = features[0]
        f_out = features[-1]
        f_ii  = features[1:-1]

        # Extracting concatenated information (CI)
        concatenated_features = self.memory_module.select(features=f_ii)
        # concatenated_features[0].shape - Tensor [N, 128, 64, 64]
        # concatenated_features[1].shape - Tensor [N, 256, 32, 32]
        # concatenated_features[2].shape - Tensor [N, 512, 16, 16]

        # Multi-scale Feature Fusion (MSFF) Module
        msff_features = self.msff(features=concatenated_features)

        # Decoding Module
        predicted_mask = self.decoder(
            encoder_output=f_out,
            concat_features=[f_in] + msff_features
        )

        return predicted_mask

