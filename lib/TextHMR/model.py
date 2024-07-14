import os
import torch
import torch.nn as nn
import pandas as pd

from .regressor import Regressor
from .transformer import Transformer
from lib.dataset._motion_dataset import read_pkl
from lib.pre_train.model import Model as pre_trained_model

# ========= Text embedding ========= #
data_root = '/mnt/SKY/AMASS_proc/processed_16frames/'
# text_candidate = pd.read_csv(os.path.join(data_root, 'total_description.csv'), header=None)
# text_candidate = list(text_candidate[0][1:])
text_embeds = read_pkl(os.path.join(data_root, 'total_description_embedding.pkl'))

class Model(nn.Module):
    def __init__(self, num_total_motion) :
        super().__init__()
        self.proj_img = nn.Linear(2048, 512)
        self.proj_joint = nn.Linear(256, 32)

        self.temp_encoder = Transformer(depth=3, embed_dim=512, mlp_hidden_dim=1024, 
                                        h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=16)
        self.pre_trained_model = pre_trained_model(num_total_motion)
        self.regressor = Regressor(512+32*17)

    def forward(self, f_img, pose_2d, is_train=False, J_regressor=None):
        """
        pose_2d     : [B, T, J, 3]
        img_feat    : [B, T, 2048]
        """
        B, T = f_img.shape[:2]

        joint_feat = self.pre_trained_model.extraction_features(pose_2d, text_embeds)   # [B, T, J, dim]
        joint_feat = self.proj_joint(joint_feat)            # [B, T, J, 32]
        joint_feat = joint_feat.flatten(-2)                 # [B, T, 32*J]

        img_feat = self.proj_img(f_img)
        img_feat = self.temp_encoder(img_feat)
        f = torch.cat([joint_feat, img_feat], dim=-1)       # [B, T, 512+32*J]

        smpl_output = self.regressor(f, is_train=is_train, J_regressor=J_regressor)

        scores = None
        if not is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, -1)
                s['verts'] = s['verts'].reshape(B, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)
                s['scores'] = scores

        else:
            size = 16
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)
                s['verts'] = s['verts'].reshape(B, size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)
                s['scores'] = scores

        return smpl_output