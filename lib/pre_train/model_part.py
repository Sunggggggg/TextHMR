import torch
import torch.nn as nn
#from transformer import CrossAttention, Transformer

coco_head_idx = [0, 1, 2, 3, 4]
coco_left_arm_idx = [5, 7, 9]
coco_right_arm_idx = [6, 8, 10]
coco_left_leg_idx = [11, 13, 15]
coco_right_leg_idx = [12, 14, 16]

class PartAttention(nn.Module):
    def __init__(self, 
                 data_type='coco', 
                 seqlen=16,
                 embed_dim=256):
        super().__init__()
        self.proj_input = nn.Linear(3, embed_dim)
        # Part
        self.head_idx = eval(f'{data_type}_head_idx')
        self.left_arm_idx = eval(f'{data_type}_left_arm_idx')
        self.right_arm_idx = eval(f'{data_type}_right_arm_idx')
        self.left_leg_idx = eval(f'{data_type}_left_leg_idx')
        self.right_leg_idx = eval(f'{data_type}_right_leg_idx')

        self.head_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//2,
                                  kernel_size=(3, 5), padding=(1, 2))
        self.left_arm_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//2,
                                  kernel_size=(3, 3), padding=(1, 1))
        self.right_arm_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//2,
                                  kernel_size=(3, 3), padding=(1, 1))
        self.left_leg_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//2,
                                  kernel_size=(3, 3), padding=(1, 1))
        self.right_leg_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//2,
                                  kernel_size=(3, 3), padding=(1, 1))
        
        self.proj_dec = nn.Linear(embed_dim, embed_dim//2)
        self.head_dec = nn.Conv1d(in_channels=10, out_channels=10//2, kernel_size=1)
        self.left_arm_dec = nn.Conv1d(in_channels=6, out_channels=6//2, kernel_size=1)
        self.right_arm_dec = nn.Conv1d(in_channels=6, out_channels=6//2, kernel_size=1)
        self.left_leg_dec = nn.Conv1d(in_channels=6, out_channels=6//2, kernel_size=1)
        self.right_leg_dec = nn.Conv1d(in_channels=6, out_channels=6//2, kernel_size=1)

        self.head_head = nn.Sequential(nn.LayerNorm(embed_dim//2), nn.Linear(embed_dim//2, 3))
        self.head_left_arm = nn.Sequential(nn.LayerNorm(embed_dim//2), nn.Linear(embed_dim//2, 3))
        self.head_right_arm = nn.Sequential(nn.LayerNorm(embed_dim//2), nn.Linear(embed_dim//2, 3))
        self.head_left_leg = nn.Sequential(nn.LayerNorm(embed_dim//2), nn.Linear(embed_dim//2, 3))
        self.head_right_leg = nn.Sequential(nn.LayerNorm(embed_dim//2), nn.Linear(embed_dim//2, 3))

    def forward(self, pose):
        """
        pose : [B, T, J, c]
        """
        B, T = pose.shape[:2]
        full_body = self.proj_input(pose)

        head = full_body[:, :, self.head_idx]            # [B, T, 5, 3]
        left_arm = full_body[:, :, self.left_arm_idx]    # [B, T, 3, 3]
        right_arm = full_body[:, :, self.right_arm_idx]  # [B, T, 3, 3]
        left_leg = full_body[:, :, self.left_leg_idx]    # [B, T, 3, 3]
        right_leg = full_body[:, :, self.right_leg_idx]  # [B, T, 3, 3]

        # head
        enc_head = self.head_conv(head.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)     # [B, T, 5, c]
        enc_left_arm = self.left_arm_conv(left_arm.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)     # [B, T, 3, c]
        enc_right_arm = self.right_arm_conv(right_arm.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)  # [B, T, 3, c]
        enc_left_leg = self.left_leg_conv(left_leg.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)     # [B, T, 3, c]
        enc_right_leg = self.right_leg_conv(right_leg.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)  # [B, T, 3, c]

        # Concat 
        full_body = self.proj_dec(full_body)
        proj_head = full_body[:, :, self.head_idx]            # [B, T, 5, c]
        proj_left_arm = full_body[:, :, self.left_arm_idx]    # [B, T, 3, c]
        proj_right_arm = full_body[:, :, self.right_arm_idx]  # [B, T, 3, c]
        proj_left_leg = full_body[:, :, self.left_leg_idx]    # [B, T, 3, c]
        proj_right_leg = full_body[:, :, self.right_leg_idx]  # [B, T, 3, c]

        enc_head = torch.cat([proj_head, enc_head], dim=2)                 # [B, T, 3+3, c]
        enc_left_arm = torch.cat([proj_left_arm, enc_left_arm], dim=2)     # [B, T, 3+3, c]
        enc_right_arm = torch.cat([proj_right_arm, enc_right_arm], dim=2)  # [B, T, 3+3, c]
        enc_left_leg = torch.cat([proj_left_leg, enc_left_leg], dim=2)     # [B, T, 3+3, c]
        enc_right_leg = torch.cat([proj_right_leg, enc_right_leg], dim=2)  # [B, T, 3+3, c]
        
        dec_head = self.head_dec(enc_head.reshape(B*T, 10, -1)).reshape(B, T, 5, -1)                     # [B, T, 5, c]
        dec_left_arm = self.left_arm_dec(enc_left_arm.reshape(B*T, 6, -1)).reshape(B, T, 3, -1)         # [B, T, 3, c]
        dec_right_arm = self.right_arm_dec(enc_right_arm.reshape(B*T, 6, -1)).reshape(B, T, 3, -1)      # [B, T, 3, c]
        dec_left_leg = self.left_leg_dec(enc_left_leg.reshape(B*T, 6, -1)).reshape(B, T, 3, -1)         # [B, T, 3, c]
        dec_right_leg = self.right_leg_dec(enc_right_leg.reshape(B*T, 6, -1)).reshape(B, T, 3, -1)      # [B, T, 3, c]

        pose[:, :, self.head_idx] += self.head_head(dec_head)
        pose[:, :, self.left_arm_idx] += self.head_left_arm(dec_left_arm)
        pose[:, :, self.right_arm_idx] += self.head_right_arm(dec_right_arm)
        pose[:, :, self.left_leg_idx] += self.head_left_leg(dec_left_leg)
        pose[:, :, self.right_leg_idx] += self.head_right_leg(dec_right_leg)
        
        return pose
    
pose2d = torch.rand((4, 16, 17, 3))
model = PartAttention('coco')

model(pose2d).shape
