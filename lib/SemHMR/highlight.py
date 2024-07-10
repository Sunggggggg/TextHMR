import torch
import torch.nn as nn

def softmax_with_temperature(x, beta=0.02, d=1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

class Highlighter(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.proj_unbaised1 = nn.Linear(embed_dim, embed_dim)
        self.proj_unbaised2 = nn.Linear(embed_dim, embed_dim)
        self.proj_unbaised3 = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, temp_feat, text_embed):
        """
        temp_feat       : [B, T, dim]
        text_embed      : [B, N, dim]
        """
        T = temp_feat.shape[1]
        img_feat = self.proj_unbaised1(temp_feat)
        text_feat = self.proj_unbaised2(text_embed)

        matrix = img_feat @ text_feat.permute(0, 2, 1)      # [B, T, N]
        softargmax = softmax_with_temperature(matrix, d=-1)
        matrix = nn.functional.softmax(matrix, dim=-1)      # [B, T, N]

        feat = matrix @ self.proj_unbaised3(text_embed)    # [B, N, dim]

        return feat, softargmax
     
def get_model(embed_dim=512):
    model = Highlighter(embed_dim=embed_dim)
    return model