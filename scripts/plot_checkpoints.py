#!/usr/bin/env python3
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

DATA  = Path('/home/aidouni/meb_texture_seg/output_ouassim/compare_checkpoints/results_q1.json')
OUT   = Path('/home/aidouni/meb_texture_seg/output_ouassim/compare_checkpoints')

with open(DATA) as f:
    data = json.load(f)

CKPTS = ['base', 'ft_0.3', 'ft_1.0']
CKPT_LABELS = {'base': 'Base', 'ft_0.3': 'FT η=0.3', 'ft_1.0': 'FT η=1.0'}
CKPT_COLORS = {'base': '#5B8DB8', 'ft_0.3': '#E8A838', 'ft_1.0': '#D95B43'}

TEXTURES = {'1': 'Tot.hom.', '3': 'Faisceaux', '4': 'Filaments',
            '6': 'Strat.sin', '7': 'Granuleux', '9': 'Trou'}

BLOCKS = [f'block_{i}' for i in range(16)] + [f'stage_{i}_fpn' for i in range(1, 5)]
BLOCK_LABELS = [f'b{i}' for i in range(16)] + ['s1', 's2', 's3', 's4']

# ─── Figure 1 : heatmaps côte à côte ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
fig.suptitle('Recall LP LOIO par bloc et texture — comparaison checkpoints',
             fontsize=13, y=1.01)

tex_ids = list(TEXTURES.keys())
tex_names = [TEXTURES[t] for t in tex_ids]

for ax, ck in zip(axes, CKPTS):
    mat = np.zeros((len(BLOCKS), len(tex_ids)))
    for i, blk in enumerate(BLOCKS):
        for j, tid in enumerate(tex_ids):
            v = data[ck].get(blk, {}).get(tid, {}).get('mean', np.nan)
            mat[i, j] = v if v is not None else np.nan

    im = ax.imshow(mat, cmap='RdYlGn', vmin=0.2, vmax=1.0,
                   aspect='auto', origin='upper')

    ax.set_xticks(range(len(tex_ids)))
    ax.set_xticklabels(tex_names, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(BLOCKS)))
    if ax == axes[0]:
        ax.set_yticklabels(BLOCK_LABELS, fontsize=8)
    ax.set_title(CKPT_LABELS[ck], fontsize=11, fontweight='bold', pad=8)

    # Valeurs dans les cellules
    for i in range(len(BLOCKS)):
        for j in range(len(tex_ids)):
            v = mat[i, j]
            if not np.isnan(v):
                color = 'white' if v < 0.45 or v > 0.82 else '#222'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6.5, color=color)

    # Séparateur blocs / FPN
    ax.axhline(15.5, color='white', lw=1.5, ls='--', alpha=0.7)
    ax.text(-0.6, 7.5, 'blocs\ntrunk', va='center', ha='right',
            fontsize=7, color='#555', rotation=90)
    ax.text(-0.6, 17.5, 'FPN', va='center', ha='right',
            fontsize=7, color='#555', rotation=90)

plt.colorbar(im, ax=axes[-1], label='Recall', fraction=0.04, pad=0.04)
plt.tight_layout()
plt.savefig(OUT / 'heatmap_checkpoints.png', dpi=150, bbox_inches='tight')
plt.close()
print("Heatmap sauvée")

# ─── Figure 2 : courbes recall vs bloc, une par texture ───────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
fig.suptitle('Recall LP LOIO vs profondeur de bloc — par texture',
             fontsize=13, y=1.01)

x = np.arange(len(BLOCKS))

for ax, tid in zip(axes.flat, tex_ids):
    for ck in CKPTS:
        means = []
        stds  = []
        for blk in BLOCKS:
            v = data[ck].get(blk, {}).get(tid, {})
            means.append(v.get('mean', np.nan))
            stds.append(v.get('std', 0))

        means = np.array(means)
        stds  = np.array(stds)
        color = CKPT_COLORS[ck]

        ax.plot(x, means, color=color, lw=1.8, label=CKPT_LABELS[ck])
        ax.fill_between(x, means - 0.3*stds, means + 0.3*stds,
                        color=color, alpha=0.12)

    ax.axvline(15.5, color='#aaa', lw=1, ls='--')
    ax.axhline(0.5, color='#ccc', lw=0.8, ls=':')
    ax.set_title(TEXTURES[tid], fontsize=11, fontweight='bold')
    ax.set_xticks([0, 5, 10, 15, 16, 17, 18, 19])
    ax.set_xticklabels(['b0','b5','b10','b15','s1','s2','s3','s4'], fontsize=8)
    ax.set_ylim(0.1, 1.05)
    ax.grid(axis='y', alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

axes[0, 0].legend(fontsize=8, loc='lower right')
for ax in axes[:, 0]:
    ax.set_ylabel('Recall', fontsize=9)

plt.tight_layout()
plt.savefig(OUT / 'curves_checkpoints.png', dpi=150, bbox_inches='tight')
plt.close()
print("Courbes sauvées")

# ─── Figure 3 : meilleur recall par texture et par checkpoint (barres groupées)
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Meilleur recall (tous blocs) par texture et checkpoint', fontsize=12)

n_tex = len(tex_ids)
n_ck  = len(CKPTS)
w = 0.25
offsets = np.array([-w, 0, w])

for ci, ck in enumerate(CKPTS):
    bests = []
    for tid in tex_ids:
        best = max(
            (data[ck].get(blk, {}).get(tid, {}).get('mean', 0) or 0)
            for blk in BLOCKS
        )
        bests.append(best)
    ax.bar(np.arange(n_tex) + offsets[ci], bests, w - 0.02,
           color=CKPT_COLORS[ck], label=CKPT_LABELS[ck],
           edgecolor='white', linewidth=0.5)

ax.set_xticks(np.arange(n_tex))
ax.set_xticklabels(tex_names, fontsize=10)
ax.set_ylabel('Recall LOIO (max sur tous les blocs)', fontsize=10)
ax.set_ylim(0, 1.1)
ax.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig(OUT / 'best_per_texture_checkpoints.png', dpi=150, bbox_inches='tight')
plt.close()
print("Barres groupées sauvées")
