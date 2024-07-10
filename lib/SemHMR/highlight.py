import torch
import torch.nn as nn

class Highlighter(nn.Module):
    def __init__(self, embed_dim=512, stride=1):
        super().__init__()
        self.stride = stride
        self.proj_unbaised1 = nn.Linear(embed_dim, embed_dim)
        self.proj_unbaised2 = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, temp_feat, text_embed):
        """
        temp_feat       : [B, T, dim]
        text_embed      : [B, N, dim]
        """
        T = temp_feat.shape[1]
        img_feat = self.proj_unbaised1(temp_feat)
        text_feat = self.proj_unbaised2(text_embed)

        matrix = img_feat @ text_feat.permute(0, 2, 1)      # [B, T, N]
        matrix = nn.functional.softmax(matrix, dim=-1)      # [B, T, N]

        return matrix
     
def get_model(embed_dim=512, stride=1):
    model = Highlighter(embed_dim=embed_dim, stride=stride)
    return model