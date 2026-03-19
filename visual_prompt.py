import math

import torch
import torch.nn as nn


class VisualPromptAligner(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor):
        # visual_features: [B, D]
        # text_features: [B, T, D] or [1, T, D]
        if text_features.dim() != 3:
            raise ValueError(f"text_features must be 3D, got shape {tuple(text_features.shape)}")

        if visual_features.dim() != 2:
            raise ValueError(f"visual_features must be 2D, got shape {tuple(visual_features.shape)}")

        if text_features.size(0) == 1 and visual_features.size(0) > 1:
            text_features = text_features.expand(visual_features.size(0), -1, -1)
        elif text_features.size(0) != visual_features.size(0):
            raise ValueError(
                "Batch size mismatch between visual and text features: "
                f"{visual_features.size(0)} vs {text_features.size(0)}"
            )

        query = visual_features.unsqueeze(1)
        scale = math.sqrt(visual_features.size(-1))
        attention_logits = torch.matmul(query, text_features.transpose(1, 2)) / scale
        attention_weights = torch.softmax(attention_logits, dim=-1)
        visual_prompt = torch.matmul(attention_weights, text_features).squeeze(1)

        fused = torch.cat([visual_features, visual_prompt], dim=-1)
        aligned_visual = self.mlp(fused)
        aligned_visual = self.norm(aligned_visual + visual_features)
        aligned_visual = aligned_visual / aligned_visual.norm(dim=-1, keepdim=True)

        return aligned_visual, visual_prompt, attention_weights.squeeze(1)
