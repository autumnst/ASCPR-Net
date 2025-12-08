import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import weights_init
from models.generator.cfa import FEM_and_GFAM, CFA
from models.generator.bigff import BiGFF
from models.generator.pconv import PConvBNActiv
from models.generator.projection import Feature2Structure, Feature2Texture
from models.MAPs.multi_scale_prior import MultiScalePriorAggregator
from models.MAPs.prior_modulation import StructureAwareModulation


class AAI_Extractor(nn.Module):

    def __init__(self, in_channels=3, mid_channels=64, out_channels=64):
        super(AAI_Extractor, self).__init__()
        self.projection = PConvBNActiv(in_channels, mid_channels, activ='leaky')
        
        self.res_layer1 = PConvBNActiv(mid_channels, mid_channels, activ='leaky')
        self.res_layer2 = PConvBNActiv(mid_channels, mid_channels, activ='leaky')
        self.res_layer3 = PConvBNActiv(mid_channels, mid_channels, activ='leaky')
        self.res_layer4 = PConvBNActiv(mid_channels, mid_channels, activ='leaky')
    
        self.out_layer = PConvBNActiv(mid_channels, out_channels, activ='leaky')

    def forward(self, image, mask):
        image, mask = self.projection(image, mask)
        aai_1, mask1 = self.res_layer1(image, mask)
        aai_1 += image 
        aai_2, mask2 = self.res_layer2(aai_1, mask1)
        aai_2 += aai_1
        aai_3, mask3 = self.res_layer3(aai_2, mask2)
        aai_3 += aai_2
        aai_4, mask4 = self.res_layer4(aai_3, mask3)
        aai_4 += aai_3
        aai, mask = self.out_layer(aai_4, mask4)
        return aai


class Extractor(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, out_channels=64):
        super(Extractor, self).__init__()
        self.aai_extractor_1 = AAI_Extractor()

    def forward(self, image, mask):
        image = self.aai_extractor_1(image, mask)
        return image.detach()


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Generator(nn.Module):

    def __init__(self, image_in_channels=3, edge_in_channels=2,
                 out_channels=3, init_weights=True, aai_channels=64):
        super(Generator, self).__init__()

        self.freeze_ec_bn = False

        # -----------------------
        # texture encoder-decoder
        # -----------------------
        self.ec_texture_1 = PConvBNActiv(image_in_channels, 64, bn=False, sample='down-7')
        self.ec_texture_2 = PConvBNActiv(64, 128, sample='down-5')
        self.ec_texture_3 = PConvBNActiv(128, 256, sample='down-5')
        self.ec_texture_4 = PConvBNActiv(256, 512, sample='down-3')
        self.ec_texture_5 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_texture_6 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_texture_7 = PConvBNActiv(512, 512, sample='down-3')

        self.dc_texture_7 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_texture_6 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_texture_5 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_texture_4 = PConvBNActiv(512 + 256, 256, activ='leaky')
        self.dc_texture_3 = PConvBNActiv(256 + 128, 128, activ='leaky')
        self.dc_texture_2 = PConvBNActiv(128 + 64, 64, activ='leaky')
        self.dc_texture_1 = PConvBNActiv(64 + out_channels, 64, activ='leaky')

        # -------------------------
        # adaptive auxiliary information encoder-decoder
        # -------------------------
        self.aai_extractor_1 = AAI_Extractor()
        self.ec_aai_1 = PConvBNActiv(aai_channels, 64, bn=False, sample='down-7')
        self.ec_aai_2 = PConvBNActiv(64, 128, sample='down-5')
        self.ec_aai_3 = PConvBNActiv(128, 256, sample='down-5')
        self.ec_aai_4 = PConvBNActiv(256, 512, sample='down-3')
        self.ec_aai_5 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_aai_6 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_aai_7 = PConvBNActiv(512, 512, sample='down-3')

        self.dc_aai_7 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_aai_6 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_aai_5 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_aai_4 = PConvBNActiv(512 + 256, 256, activ='leaky')
        self.dc_aai_3 = PConvBNActiv(256 + 128, 128, activ='leaky')
        self.dc_aai_2 = PConvBNActiv(128 + 64, 64, activ='leaky')
        self.dc_aai_1 = PConvBNActiv(64 + aai_channels, 64, activ='leaky')

        # -----------------------------------
        # Bi-directional Gated Feature Fusion
        # -----------------------------------
        # 注意：BiGFF 实际返回 128 通道特征（两个 64 特征融合后拼接）
        self.bigff = BiGFF(in_channels=64, out_channels=64)

        # ---------- MAPs-APA：三尺度结构先验 ----------
        self.struct_prior_agg = MultiScalePriorAggregator(
            feat_channels=64,
            prior_channels=(64, 64, 64),
            out_channels=64
        )
        self.struct_mod = StructureAwareModulation(
            feat_channels=64,
            prior_channels=64,
            out_channels=64
        )

        # ------------------------------
        # Contextual Feature Aggregation
        # ------------------------------
        # BiGFF 输出 128 通道 → 压回 64 通道
        self.fusion_layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.cfa_texture = FEM_and_GFAM(nc=3, n_fem_res=1, n_heads=4, ksize=8, stride_1=8, stride_2=8)
        self.cfa_aai = FEM_and_GFAM(nc=64, n_fem_res=1, n_heads=4, ksize=8, stride_1=8, stride_2=8)

        # 目前没用到，可以保留做后续实验
        self.fusion_layer4 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # out_layer：输入为 output_2(64) + output_bigff_2(128) = 192 通道
        self.out_layer = nn.Sequential(
            nn.Conv2d(192, 3, kernel_size=1),
            nn.Tanh()
        )

        if init_weights:
            self.apply(weights_init())


    def forward(self, input_image, input_edge, mask, input_aai, is_train,
                struct_priors=None):

        # -----------------------
        # AAI 分支输入预处理
        # -----------------------
        if is_train:
            aai_feature = self.aai_extractor_1(input_aai * mask, mask.expand_as(input_aai))
            aai_input = aai_feature
            aai_mask = mask.expand_as(aai_feature)
        else:
            aai_feature = input_aai
            aai_mask = mask.expand_as(aai_feature)
            aai_input = aai_feature

        ec_textures = {}
        ec_aai = {}

        # -----------------------
        # 纹理分支编码
        # -----------------------
        input_texture_mask = torch.cat((mask, mask, mask), dim=1)
        ec_textures['ec_t_0'], ec_textures['ec_t_masks_0'] = input_image, input_texture_mask
        ec_textures['ec_t_0'] = self.cfa_texture(ec_textures['ec_t_0'])

        ec_textures['ec_t_1'], ec_textures['ec_t_masks_1'] = self.ec_texture_1(
            ec_textures['ec_t_0'], ec_textures['ec_t_masks_0'])
        ec_textures['ec_t_2'], ec_textures['ec_t_masks_2'] = self.ec_texture_2(
            ec_textures['ec_t_1'], ec_textures['ec_t_masks_1'])
        ec_textures['ec_t_3'], ec_textures['ec_t_masks_3'] = self.ec_texture_3(
            ec_textures['ec_t_2'], ec_textures['ec_t_masks_2'])
        ec_textures['ec_t_4'], ec_textures['ec_t_masks_4'] = self.ec_texture_4(
            ec_textures['ec_t_3'], ec_textures['ec_t_masks_3'])
        ec_textures['ec_t_5'], ec_textures['ec_t_masks_5'] = self.ec_texture_5(
            ec_textures['ec_t_4'], ec_textures['ec_t_masks_4'])
        ec_textures['ec_t_6'], ec_textures['ec_t_masks_6'] = self.ec_texture_6(
            ec_textures['ec_t_5'], ec_textures['ec_t_masks_5'])
        ec_textures['ec_t_7'], ec_textures['ec_t_masks_7'] = self.ec_texture_7(
            ec_textures['ec_t_6'], ec_textures['ec_t_masks_6'])

        # -----------------------
        # AAI 分支编码
        # -----------------------
        ec_aai['ec_a_0'], ec_aai['ec_a_masks_0'] = aai_input, aai_mask
        ec_aai['ec_a_0'] = self.cfa_aai(ec_aai['ec_a_0'])

        ec_aai['ec_a_1'], ec_aai['ec_a_masks_1'] = self.ec_aai_1(
            ec_aai['ec_a_0'], ec_aai['ec_a_masks_0'])
        ec_aai['ec_a_2'], ec_aai['ec_a_masks_2'] = self.ec_aai_2(
            ec_aai['ec_a_1'], ec_aai['ec_a_masks_1'])
        ec_aai['ec_a_3'], ec_aai['ec_a_masks_3'] = self.ec_aai_3(
            ec_aai['ec_a_2'], ec_aai['ec_a_masks_2'])
        ec_aai['ec_a_4'], ec_aai['ec_a_masks_4'] = self.ec_aai_4(
            ec_aai['ec_a_3'], ec_aai['ec_a_masks_3'])
        ec_aai['ec_a_5'], ec_aai['ec_a_masks_5'] = self.ec_aai_5(
            ec_aai['ec_a_4'], ec_aai['ec_a_masks_4'])
        ec_aai['ec_a_6'], ec_aai['ec_a_masks_6'] = self.ec_aai_6(
            ec_aai['ec_a_5'], ec_aai['ec_a_masks_5'])
        ec_aai['ec_a_7'], ec_aai['ec_a_masks_7'] = self.ec_aai_7(
            ec_aai['ec_a_6'], ec_aai['ec_a_masks_6'])

        # -----------------------
        # 纹理分支解码
        # -----------------------
        dc_texture_out, dc_texture_mask = ec_textures['ec_t_7'], ec_textures['ec_t_masks_7']
        dc_texture = F.interpolate(dc_texture_out, scale_factor=2, mode='bilinear')
        dc_texture_mask = F.interpolate(dc_texture_mask, scale_factor=2, mode='nearest')

        for _ in range(6, 0, -1):
            ec_texture_skip = f'ec_t_{_ - 1}'
            ec_texture_masks_skip = f'ec_t_masks_{_ - 1}'
            dc_conv = f'dc_texture_{_}'

            dc_texture = F.interpolate(dc_texture, scale_factor=2, mode='bilinear')
            dc_texture_mask = F.interpolate(dc_texture_mask, scale_factor=2, mode='nearest')

            dc_texture = torch.cat((dc_texture, ec_textures[ec_texture_skip]), dim=1)
            dc_texture_mask = torch.cat((dc_texture_mask, ec_textures[ec_texture_masks_skip]), dim=1)

            dc_texture, dc_texture_mask = getattr(self, dc_conv)(dc_texture, dc_texture_mask)

        # ---------- 使用多尺度结构先验进行融合 ----------
        if struct_priors is not None:
            prior_map = self.struct_prior_agg(
                dc_texture,
                [struct_priors['s1'], struct_priors['s2'], struct_priors['s3']]
            )
            dc_texture = self.struct_mod(dc_texture, prior_map)

        # -----------------------
        # AAI 分支解码
        # -----------------------
        dc_aai, dc_aai_masks = ec_aai['ec_a_7'], ec_aai['ec_a_masks_7']

        for _ in range(7, 0, -1):
            ec_aai_skip = f'ec_a_{_ - 1}'
            ec_aai_masks_skip = f'ec_a_masks_{_ - 1}'
            dc_conv = f'dc_aai_{_}'

            dc_aai = F.interpolate(dc_aai, scale_factor=2, mode='bilinear')
            dc_aai_masks = F.interpolate(dc_aai_masks, scale_factor=2, mode='nearest')

            dc_aai = torch.cat((dc_aai, ec_aai[ec_aai_skip]), dim=1)
            dc_aai_masks = torch.cat((dc_aai_masks, ec_aai[ec_aai_masks_skip]), dim=1)

            dc_aai, dc_aai_masks = getattr(self, dc_conv)(dc_aai, dc_aai_masks)

        # -------------------
        # BiGFF + 融合 & 输出
        # -------------------
        projected_image = 0  # 保持原 APLRL 接口格式

        # BiGFF 融合：输入两个 64 通道特征 → 输出 128 通道
        output_bigff_2 = self.bigff(dc_texture, dc_aai)      # [B, 128, H, W]
        # 再用 fusion_layer1 压缩到 64 通道
        output_2 = self.fusion_layer1(output_bigff_2)        # [B, 64, H, W]

        # 最终输出：拼接 64 + 128 = 192 通道
        output = self.out_layer(torch.cat((output_2, output_bigff_2), dim=1))

        # dc_aai 作为 AAI latent 返回
        return output, projected_image, aai_feature.detach(), dc_aai

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_ec_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
