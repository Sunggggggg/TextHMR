import os
import torch
import torch.nn as nn
from lib.core.config import BASE_DATA_DIR

from .lifter import add_joint

import lib.models.lifter as lifter
import lib.models.temporal_encoder as temporal_encoder
import lib.models.text_encoder as text_encoder
import lib.models.linker as linker

class Model(nn.Module):
    def __init__(self, 
                 seqlen=16,
                 num_joints=19, 
                 embed_dim=256, 
                 depth=3,
                 lifter_pretrained=os.path.join(BASE_DATA_DIR, 'pose_3dpw.pth.tar')
                 ) :
        super().__init__()
        self.pose_lifter = lifter.get_model(num_joints, embed_dim, depth, lifter_pretrained)
        self.init_hmr = temporal_encoder.get_model(depth=depth, length=seqlen, embed_dim=512)
        self.text_encoder = text_encoder.get_model(length=36, embed_dim=512)
        self.linking = linker.get_model(embed_dim=256)

        self.proj_text = nn.Linear(512, embed_dim)

    def forward(self, input_text, img_feat, pose2d, is_train=False, J_regressor=None):
        """
        input_text : [B, 36, 768]
        """
        B = input_text.shape[0]

        pose2d = add_joint(pose2d[..., :2])
        # First stage
        pose3d = self.pose_lifter(pose2d, img_feat)                 # [B, J, 3]
        init_theta = self.init_hmr(img_feat, is_train, J_regressor) # 
        text_embed = self.text_encoder(input_text)                  # [B, N, ]

        # Second stage
        text_embed = self.proj_text(text_embed)
        joint_guide, semantic_guide = self.linking(pose3d, text_embed)

        if is_train:
            size = 1
            for s in init_theta:
                s['theta'] = s['theta'].reshape(B, size, -1)
                s['verts'] = s['verts'].reshape(B, size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)
        else :
            for s in init_theta:
                s['theta'] = s['theta'].reshape(B, -1)
                s['verts'] = s['verts'].reshape(B, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)

        return pose3d, init_theta, joint_guide
