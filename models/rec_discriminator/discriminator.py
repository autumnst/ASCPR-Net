import torch
import torch.nn as nn

from models.rec_discriminator.structure_branch import EdgeDetector, StructureBranch
from models.rec_discriminator.texture_branch import TextureBranch


class Discriminator(nn.Module):

    def __init__(self, image_in_channels=3):
        super(Discriminator, self).__init__()

        # 纹理分支
        self.texture_branch = TextureBranch(in_channels=image_in_channels)

        # 结构分支
        self.structure_branch = StructureBranch(in_channels=image_in_channels)

        # 边缘检测
        self.edge_detector = EdgeDetector()

    def forward(self, output):

        # 自动提取结构 edge map
        edges = self.edge_detector(output)
        edges = torch.cat([edges, edges, edges], dim=1)


        # 两个判别器分支
        texture_pred = self.texture_branch(output)
        structure_pred = self.structure_branch(edges)

        # 融合（可换权重）
        return (texture_pred + structure_pred) / 2
