# ===========================
# Corrected Reconstructor
# Multi-scale Prior Output
# ===========================

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import weights_init
from models.generator.pconv import PConvBNActiv
from models.generator.projection import Feature2Texture


class Reconstructor(nn.Module):

    def __init__(self, image_in_channels=3, out_channels=3, init_weights=True):
        super(Reconstructor, self).__init__()

        self.freeze_ec_bn = False

        # -----------------------
        # Encoder (7 layers)
        # -----------------------
        self.ec_texture_1 = PConvBNActiv(image_in_channels, 64, bn=False, sample='down-7')
        self.ec_texture_2 = PConvBNActiv(64, 128, sample='down-5')
        self.ec_texture_3 = PConvBNActiv(128, 256, sample='down-5')
        self.ec_texture_4 = PConvBNActiv(256, 512, sample='down-3')
        self.ec_texture_5 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_texture_6 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_texture_7 = PConvBNActiv(512, 512, sample='down-3')

        # -----------------------
        # Decoder (7 → 1)
        # -----------------------
        self.dc_texture_7 = PConvBNActiv(512 , 512, activ='leaky')
        self.dc_texture_6 = PConvBNActiv(512 , 512, activ='leaky')
        self.dc_texture_5 = PConvBNActiv(512 , 512, activ='leaky')
        self.dc_texture_4 = PConvBNActiv(512 , 256, activ='leaky')
        self.dc_texture_3 = PConvBNActiv(256 , 128, activ='leaky')
        self.dc_texture_2 = PConvBNActiv(128 , 64 , activ='leaky')
        self.dc_texture_1 = PConvBNActiv(64  , 64 , activ='leaky')

        # -------------------
        # Feature Projection
        # -------------------
        self.texture_feature_projection = Feature2Texture()

        # =============================================================
        # Multi-scale Prior Projection (S1/S2/S3 → 统一 64 channels)
        # =============================================================
        self.prior_proj_3 = nn.Conv2d(256, 64, 1)  # ec_t_3
        self.prior_proj_4 = nn.Conv2d(512, 64, 1)  # ec_t_4
        self.prior_proj_5 = nn.Conv2d(512, 64, 1)  # ec_t_5

        # -----------------------
        # Final Output
        # -----------------------
        self.out_layer = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Tanh()
        )

        if init_weights:
            self.apply(weights_init())


    def forward(self, input_image, mask, return_priors=False):

        ec_textures = {}

        # Mask propagation
        #input_texture_mask = mask
        input_texture_mask = torch.cat((mask, mask, mask), dim=1)

        ec_textures['ec_t_0'], ec_textures['ec_t_masks_0'] = input_image, input_texture_mask

        # --- Encode ---
        for i in range(1, 8):
            ec_block = getattr(self, f'ec_texture_{i}')
            prev = f'ec_t_{i-1}'
            prev_mask = f'ec_t_masks_{i-1}'
            ec_textures[f'ec_t_{i}'], ec_textures[f'ec_t_masks_{i}'] = \
                ec_block(ec_textures[prev], ec_textures[prev_mask])

        # --- Decode ---
        dc_texture = ec_textures['ec_t_7']
        dc_mask = ec_textures['ec_t_masks_7']

        dc_texture = F.interpolate(dc_texture, scale_factor=2, mode='bilinear')
        dc_mask = F.interpolate(dc_mask, scale_factor=2, mode='nearest')
        dc_texture, dc_mask = self.dc_texture_7(dc_texture, dc_mask)

        for idx in range(6, 0, -1):
            dc_texture = F.interpolate(dc_texture, scale_factor=2, mode='bilinear')
            dc_mask = F.interpolate(dc_mask, scale_factor=2, mode='nearest')
            dc_block = getattr(self, f'dc_texture_{idx}')
            dc_texture, dc_mask = dc_block(dc_texture, dc_mask)

        # ------------------------------------
        # Texture projection (standard APLRL)
        # ------------------------------------
        projected_image = self.texture_feature_projection(dc_texture)

        # -----------------------
        # Final output image
        # -----------------------
        output = self.out_layer(dc_texture)

        # ============================================================
        #          【核心】三尺度结构先验输出 (S1/S2/S3)
        # ============================================================
        if return_priors:

            S1 = self.prior_proj_3(ec_textures['ec_t_3'])  # high-frequency structure
            S2 = self.prior_proj_4(ec_textures['ec_t_4'])  # mid-level structure
            S3 = self.prior_proj_5(ec_textures['ec_t_5'])  # global structure
            # ======================================================
            # 统一多尺度先验的空间分辨率，让它们进入 APA 模块
            # ======================================================
            # 假设 decoder 输出尺寸为 H×W
            target_h, target_w = dc_texture.shape[2], dc_texture.shape[3]

            S1 = F.interpolate(S1, size=(target_h, target_w), mode='bilinear', align_corners=False)
            S2 = F.interpolate(S2, size=(target_h, target_w), mode='bilinear', align_corners=False)
            S3 = F.interpolate(S3, size=(target_h, target_w), mode='bilinear', align_corners=False)

            priors = {
                's1': S1,
                's2': S2,
                's3': S3
            }

            return output, projected_image, dc_texture, priors

        return output, projected_image, dc_texture


    def train(self, mode=True):
        super().train(mode)
        if self.freeze_ec_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
