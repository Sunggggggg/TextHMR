import torch
import torch.nn as nn

def normalization(x):
    """ x : [T, N] """
    x_min = torch.min(x, dim=-1, keepdim=True)[0]
    x -= x_min
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    x_max[x_max == 0] += 1e-30
    x /= x_max
    return x

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
        self.num_select = 4
        self.proj_unbaised1 = nn.Linear(embed_dim, embed_dim)
        self.proj_unbaised2 = nn.Linear(embed_dim, embed_dim)
        self.proj_unbaised3 = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, temp_feat, text_embed, caption_len):
        """
        temp_feat       : [B, T, dim]
        text_embed      : [B, N, dim]
        caption_len     : [B]

        return
        selected_text_embeds    : [B, self.num_select, dim]
        loss
        """
        B, T = temp_feat.shape[:2]
        img_feat = self.proj_unbaised1(temp_feat)           # One-hot vector
        text_feat = self.proj_unbaised2(text_embed)

        matrix = img_feat @ text_feat.permute(0, 2, 1)      # [B, T, N]

        batch_matrixs = []
        selected_text_embeds = []
        for b in range(B) :
            # Remove padding
            batch_matrix = matrix[b, :, :caption_len[b]]              # [T, n]
            norm_matrix = normalization(batch_matrix)                 # 
            
            # Selection
            mid_frame = norm_matrix[T//2] 
            _, indices = torch.sort(mid_frame, dim=-1, descending=True)         # [n]
            batch_text_embed = text_embed[b, :caption_len[b]]                   # [n, dim]
            selected_text_embed = batch_text_embed[indices[:self.num_select]]   # [4, dim]
            selected_text_embeds.append(selected_text_embed)

            # Loss
            batch_matrixs.append(batch_matrix)

        selected_text_embeds = torch.stack(selected_text_embeds, dim=0)         # [B, n, dim]
        return selected_text_embeds, batch_matrixs
     
def get_model(embed_dim=512):
    model = Highlighter(embed_dim=embed_dim)
    return model