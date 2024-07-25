import torch
import torch.nn as nn
from .motion_encoder import GraphormerNet as motionencoder
from .regressor import Regressor
from .operation.transformer import Transformer

class Model(nn.Module):
    def __init__(self,
                 seqlen,
                 num_total_motion,
                 text_archive,
                 pretrained,
                 pretrained_freeze=True
                 ) :
        super().__init__()
        self.seqlen = seqlen
        self.text_archive = text_archive

        self.motion_encoder = motionencoder(num_frames=64, num_joints=17, embed_dim=256, depth=10)
        
        self.proj_enc = nn.Linear(2048, 512)
        self.temp_encoder = Transformer(depth=3, embed_dim=512, mlp_hidden_dim=512*4., 
                                        h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=16)
        
        self.proj_joint = nn.Linear(256, 32)
        self.gru = nn.GRU(input_size=512+32*17, hidden_size=1024, bidirectional=True, num_layers=2)
        self.proj_dec = nn.Linear(2048, 256)
        self.regressor = Regressor(256)

        if pretrained :
            pretrained_dict = torch.load(pretrained)['gen_state_dict']

            self.motion_encoder.load_state_dict(pretrained_dict)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

        if pretrained_freeze :
            self.motion_encoder.eval()
            for p in self.motion_encoder.parameters():
                p.requires_grad = False

    def forward(self, f_img, pose_2d, is_train=False, J_regressor=None):
        
        B, T, J = pose_2d.shape[:3]
        start, end = (T//2-8), (T//2+8) 
        # Motion feat.
        motion_feat = self.motion_encoder(pose_2d)  # [B, T, J, dim]
        motion_feat = motion_feat[:, start:end]     # [B, 16, J, dim]
        motion_feat = self.proj_joint(motion_feat)  # [B, 16, J, 32]
        motion_feat = motion_feat.flatten(-2)
        
        # Image feat.
        f_img = f_img[:, start:end]                 # [B, 16, 2048]
        img_feat = self.proj_enc(f_img)                     
        img_feat = self.temp_encoder(img_feat)      # [B, T, 512] 

        # 
        feat = torch.cat([motion_feat, img_feat], dim=-1)
        feat, _ = self.gru(feat)
        feat = self.proj_dec(feat)
        
        if is_train:
            feat = feat
        else :
            feat = feat[:, 16//2][:, None]

        smpl_output, _ = self.regressor(feat, is_train=is_train, J_regressor=J_regressor)

        if not is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, -1)
                s['verts'] = s['verts'].reshape(B, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)

        else:
            size = 16
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)
                s['verts'] = s['verts'].reshape(B, size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)

        return smpl_output, None