import torch
import torch.nn as nn
from transformer import Attention
from einops import rearrange

coco_head_idx = [0, 1, 2, 3, 4]
coco_left_arm_idx = [5, 7, 9]
coco_right_arm_idx = [6, 8, 10]
coco_left_leg_idx = [11, 13, 15]
coco_right_leg_idx = [12, 14, 16]

class PartAttentionModule(nn.Module):
    def __init__(self, 
                 data_type='coco',
                 embed_dim=256):
        super().__init__()
        # Part
        self.head_idx = eval(f'{data_type}_head_idx')
        self.left_arm_idx = eval(f'{data_type}_left_arm_idx')
        self.right_arm_idx = eval(f'{data_type}_right_arm_idx')
        self.left_leg_idx = eval(f'{data_type}_left_leg_idx')
        self.right_leg_idx = eval(f'{data_type}_right_leg_idx')

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(), nn.LayerNorm(embed_dim)
            )
        self.left_arm_conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(), nn.LayerNorm(embed_dim)
            )
        self.right_arm_conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(), nn.LayerNorm(embed_dim)
            )
        self.left_leg_conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(), nn.LayerNorm(embed_dim)
            )
        self.right_leg_conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(), nn.LayerNorm(embed_dim)
            )

    def forward(self, full_body):
        """
        full_body : [B, T, J, dim]
        """
        B, T = full_body.shape[:2]
        head = full_body[:, :, self.head_idx]            # [B, T, 5, 256]
        left_arm = full_body[:, :, self.left_arm_idx]    # [B, T, 3, 256]
        right_arm = full_body[:, :, self.right_arm_idx]  # [B, T, 3, 256]
        left_leg = full_body[:, :, self.left_leg_idx]    # [B, T, 3, 256]
        right_leg = full_body[:, :, self.right_leg_idx]  # [B, T, 3, 256]

        # head
        enc_head = self.head_conv(head.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)                 # [B, T, 5, 256]
        enc_left_arm = self.left_arm_conv(left_arm.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)     # [B, T, 3, 256]
        enc_right_arm = self.right_arm_conv(right_arm.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)  # [B, T, 3, 256]
        enc_left_leg = self.left_leg_conv(left_leg.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)     # [B, T, 3, 256]
        enc_right_leg = self.right_leg_conv(right_leg.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)  # [B, T, 3, 256]

        # Skip
        full_body[:, :, self.head_idx] = head + enc_head
        full_body[:, :, self.left_arm_idx] = left_arm + enc_left_arm
        full_body[:, :, self.right_arm_idx] = right_arm + enc_right_arm
        full_body[:, :, self.left_leg_idx] = left_leg + enc_left_leg
        full_body[:, :, self.right_leg_idx] = right_leg + enc_right_leg
        
        return full_body

class 

class PartAttention(nn.Module):
    def __init__(self, data_type='coco', depth=4, embed_dim=256, num_joints=17, num_frames=16) :
        super().__init__()
        self.depth = depth
        self.proj_input = nn.Linear(2, embed_dim)

        # Part
        self.head_idx = eval(f'{data_type}_head_idx')
        self.left_arm_idx = eval(f'{data_type}_left_arm_idx')
        self.right_arm_idx = eval(f'{data_type}_right_arm_idx')
        self.left_leg_idx = eval(f'{data_type}_left_leg_idx')
        self.right_leg_idx = eval(f'{data_type}_right_leg_idx')

        # Encoder 
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.part_atten = nn.ModuleList(
            PartAttentionModule(embed_dim=embed_dim)
            for _ in range(depth)
        )
        self.spatial_atten = nn.ModuleList(
            Attention(embed_dim, num_heads=8, attn_drop=0.2, proj_drop=0.1)
            for _ in range(depth)
        )

        self.temporal_atten = nn.ModuleList(
            Attention(embed_dim, num_heads=8, attn_drop=0.2, proj_drop=0.1)
            for _ in range(depth)
        )

        self.norm_s = nn.LayerNorm(embed_dim)
        self.norm_t = nn.LayerNorm(embed_dim)

        # Joint head
        self.head_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
        self.head_left_arm = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
        self.head_right_arm = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
        self.head_left_leg = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
        self.head_right_leg = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
    
    def output_head(self, refine_body):
        refine_body[:, :, self.head_idx] = self.head_head(dec_head)
        refine_body[:, :, self.left_arm_idx] = self.head_left_arm(dec_left_arm)
        refine_body[:, :, self.right_arm_idx] = self.head_right_arm(dec_right_arm)
        refine_body[:, :, self.left_leg_idx] = self.head_left_leg(dec_left_leg)
        refine_body[:, :, self.right_leg_idx] = self.head_right_leg(dec_right_leg)

        return refine_body

    def forward(self, pose_2d):
        """
        pose_2d : [B, T, J, 2]
        """
        full_body = self.proj_input(pose_2d)        # [B, T, J, 256]
        refine_body = torch.zeros_like(full_body, device=pose_2d.device)

        for idx, (part, s_attn, t_attn) in enumerate(zip(self.part_atten, self.spatial_atten, self.temporal_atten)):
            if idx == 0 :
            full_body = rearrange(full_body, 'b t j c  -> (b t) j c')
            full_body = full_body + self.spatial_pos_embed
            full_body = self.pos_drop(full_body)
            full_body = s_attn[idx](full_body)
            full_body = self.norm_s(full_body)





        

        return