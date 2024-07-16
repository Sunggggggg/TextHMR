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
        self.num_words = 36
        self.num_select = 4
        self.proj_unbaised1 = nn.Linear(embed_dim, embed_dim)
        self.proj_unbaised2 = nn.Linear(embed_dim, embed_dim)
        self.proj_unbaised3 = nn.Linear(embed_dim, embed_dim)
    
    def atten_mask(self, caption_len):
        """
        caption_len : [B]
        atten_mask : [B, N]
        """
        atten_mask = []
        for mask_len in caption_len :
            mask = torch.ones(self.num_words, device=caption_len.device)
            mask[:mask_len] = 0.
            atten_mask.append(mask.bool())
        atten_mask = torch.stack(atten_mask, dim=0)
        return atten_mask

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
        
        mask = self.atten_mask(caption_len)                 # [B, N]
        mask = mask.unsqueeze(1).expand_as(matrix)
        matrix.masked_fill_(mask, -float('inf'))
        matrix = matrix.softmax(dim=-1)                     # [B, T, N]

        idx_list = torch.sort(matrix, dim=-1, descending=True).indices  # [B, T, N]
        #selection = torch.gather(matrix, dim=-1, index=idx_list[..., :self.num_select]) # [B, T, self.num_select]

        text_embed_selection = []
        for b, idx in enumerate(idx_list[:, T//2]):
            text_embed_selection.append(text_feat[b, idx[:4]])
        text_embed_selection = torch.stack(text_embed_selection, dim=0)

        return text_embed_selection, matrix

def get_model(embed_dim=512):
    model = Highlighter(embed_dim=embed_dim)
    return model