import torch
import torch.nn as nn


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim or embed_dim

        self.visual_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, embed_dim),
            nn.Sigmoid(),
        )

        self.final_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, embed_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def fuse_visual_prompt(self, visual_features: torch.Tensor, visual_prompt_features: torch.Tensor):
        gate = self.visual_gate(torch.cat([visual_features, visual_prompt_features], dim=-1))
        fused = gate * visual_prompt_features + (1.0 - gate) * visual_features
        fused = self.norm(fused)
        fused = fused / fused.norm(dim=-1, keepdim=True)
        return fused, gate

    def fuse_cluster_feature(self, visual_features: torch.Tensor, clustered_feature: torch.Tensor):
        gate = self.final_gate(torch.cat([visual_features, clustered_feature], dim=-1))
        fused = gate * clustered_feature + (1.0 - gate) * visual_features
        fused = self.norm(fused)
        fused = fused / fused.norm(dim=-1, keepdim=True)
        return fused, gate
