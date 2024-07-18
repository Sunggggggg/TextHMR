import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, ):
        super().__init__()

    

    def text_prediction(self, joint_feat):
        """ Text predicting via joint features
        joint_feat : [B, T, J, dim]
        """
        x = joint_feat.mean(dim=1)

        x = self.text_head[0](x)               # [B, J, d]
        x = self.text_head[1](x)               # [B, J, d]
        x = x.flatten(-2)                      # [B, J*d]
        x = self.text_head[2](x)               # [B, num_total_motion]

        return x

    def forward(self, pose_2d, text_emb, caption_mask):
        """
        pose_2d      : [B, T, J, 3] z=1
        text_emb     : [B, N, 768]
        caption_mask : [B, 36]
        """
        # Stage 1.
        joint_feat = self.st_fromer(pose_2d, return_joint=False)  # [B, T, J, dim] 
        pred_text = self.text_prediction(joint_feat)              # [B, num_total_motion]

        # Stage 2.
        text_feat = self.text_encoder(text_emb, caption_mask)     # [B, N, dim]
        joint_feat = self.co_former(joint_feat, text_feat, caption_mask)
        pred_kp_3d = self.joint_head(joint_feat)                  # [B, T, J, 3] 

        return pred_text, pred_kp_3d