import os
import torch
import torch.nn as nn

import lifter
import temporal_encoder
import text_encoder

class Model(nn.Module):
    def __init__(self, 
                 seqlen,
                 num_joint, 
                 embed_dim, 
                 depth,
                 lifter_pretrained=os.path.join()
                 ) :
        super().__init__()
        self.pose_lifter = lifter.get_model(num_joint, embed_dim, depth, lifter_pretrained)
        self.init_hmr = temporal_encoder.get_model(depth=depth, length=seqlen, embed_dim=embed_dim)
        self.text_encoder = text_encoder.get_model(length=36)

    def add_joint(self, pose2d):
        pelvis = pose2d[:,:,[11,12],:2].mean(dim=2, keepdim=True)
        neck = pose2d[:,:,[5,6],:2].mean(dim=2, keepdim=True)

        return torch.cat([pose2d, pelvis, neck], dim=2)

    def forward(self, input_text, img_feat, pose2d, is_train=False, J_regressor=None):
        """
        input_text : [B, 36, 768]
        """
        # First stage
        pose3d = self.pose_lifter(pose2d, img_feat)
        init_theta = self.init_hmr(img_feat, is_train, J_regressor)
        text_embed = self.text_encoder(input_text)

        # Second stage
        
        
        

        return pose3d, init_theta
