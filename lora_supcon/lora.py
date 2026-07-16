"""
lora.py
══════════════════════════════════════════════════════════════════════════════
PHASE 2 — LoRA sur les projections qkv des MultiScaleBlock du stage 3
(blocs 4 à 13) de l'encodeur Hiera de TextureSAM.

sortie = W_gelé(x) + (alpha/r) · B(A(dropout(x)))
r=8, alpha=16, A ~ N(0, 0.02), B=0, dropout LoRA=0.1

Tout le reste de l'encodeur est gelé (requires_grad=False).
══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn

LORA_BLOCKS = list(range(4, 14))  # blocs 4 à 13 inclus (stage 3)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


class LoRALinear(nn.Module):
    """Enveloppe LoRA autour d'un nn.Linear gelé (utilisé pour qkv)."""

    def __init__(self, orig_linear: nn.Linear, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT):
        super().__init__()
        self.orig = orig_linear
        for p in self.orig.parameters():
            p.requires_grad = False

        in_dim = orig_linear.in_features
        out_dim = orig_linear.out_features
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.randn(r, in_dim) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.orig(x)
        lora = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base + self.scaling * lora


def apply_lora(encoder: nn.Module, blocks=LORA_BLOCKS) -> list:
    """
    Gèle TOUT l'encodeur (trunk + neck), puis insère LoRALinear sur
    attn.qkv des blocs `blocks` du trunk. Retourne la liste des
    LoRALinear insérés.
    """
    for p in encoder.parameters():
        p.requires_grad = False

    lora_modules = []
    for i in blocks:
        block = encoder.trunk.blocks[i]
        lora_qkv = LoRALinear(block.attn.qkv)
        block.attn.qkv = lora_qkv
        lora_modules.append(lora_qkv)
    return lora_modules


def count_trainable_params(model: nn.Module) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
