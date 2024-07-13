import torch
import torch.nn as nn
import torch.nn.functional as F

from .STFormer import STFormer
from .CrossAtten import CoTransformer
from .text_encoder import TEncoder

class Model(nn.Module):
    def __init__(self, num_total_motion) :
        super().__init__()
        self.mid_frame = 8
        self.st_fromer = STFormer(num_frames=16, num_joints=17, embed_dim=256, depth=6, num_heads=8, mlp_ratio=2., 
                 qkv_bias=True, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.2, norm_layer=None, pretrained=False)
        
        self.text_encoder = TEncoder(depth=3, embed_dim=256, mlp_hidden_dim=256*4.,
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=36)
        
        self.co_former = CoTransformer(seqlen=16, num_joints=17, num_words=36 ,embed_dim=256)
        
        self.joint_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 3)
        )

        self.text_head = nn.ModuleList([nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Dropout()),
                                         nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Dropout()),
                                         nn.Linear(32*17, num_total_motion)])
        
    def text_prediction(self, joint_feat):
        """
        joint_feat : [B, T, J, dim]
        """
        x = joint_feat.mean(dim=1)

        x = self.text_head[0](x)               # [B, J, d]
        x = self.text_head[1](x)               # [B, J, d]
        x = x.flatten(-2)                      # [B, J*d]
        x = self.text_head[2](x)               # [B, num_total_motion]
        #x = F.softmax(x, dim=-1)

        return x

    def forward(self, pose_2d, text_emb):
        # Stage 1.
        joint_feat = self.st_fromer(pose_2d, return_joint=False)  # [B, T, J, dim] 
        pred_text = self.text_prediction(joint_feat)               

        # Stage 2.
        text_feat = self.text_encoder(text_emb)                   # [B, N, dim]
        joint_feat = self.co_former(joint_feat, text_feat)
        pred_kp_3d = self.joint_head(joint_feat)                  # [B, T, J, 3] 

        return pred_text, pred_kp_3d