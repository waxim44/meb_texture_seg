#!/usr/bin/env python3
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA = Path('/home/aidouni/meb_texture_seg/output_ouassim/lp_norm_blocks/lp_results.json')
OUT  = Path('/home/aidouni/meb_texture_seg/output_ouassim/lp_norm_blocks')

with open(DATA) as f:
    data = json.load(f)

NORMS  = ['baseline', 'gamma_0.7', 'gamma_1.5', 'zscore_image']
LABELS = {'baseline': 'Baseline', 'gamma_0.7': 'γ=0.7',
          'gamma_1.5': 'γ=1.5', 'zscore_image': 'Z-score'}
COLORS = {'baseline': '#4878CF', 'gamma_0.7': '#6ACC65',
          'gamma_1.5': '#D65F5F', 'zscore_image': '#B47CC7'}

TEXTURES = {'1': 'Tot.hom.', '3': 'Faisceaux', '4': 'Filaments',
            '5': 'Strat.rect', '6': 'Strat.sin', '7': 'Granuleux', '9': 'Trou'}

P_VALS = [str(p) for p in range(3, 14)]
P_LABELS = [f'p={p}' for p in range(3, 14)]

tex_ids   = list(TEXTURES.keys())
tex_names = [TEXTURES[t] for t in tex_ids]

# ─── Figure 1 : heatmaps côte à côte ──────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
fig.suptitle('Recall LP LOIO (test) par norme Lp et texture — comparaison normalisations',
             fontsize=12, y=1.01)

for ax, norm in zip(axes, NORMS):
    mat = np.zeros((len(P_VALS), len(tex_ids)))
    for i, p in enumerate(P_VALS):
        for j, tid in enumerate(tex_ids):
            v = data[norm].get(p, {}).get(tid, {}).get('test', np.nan)
            mat[i, j] = v if v is not None else np.nan

    im = ax.imshow(mat, cmap='RdYlGn', vmin=0.0, vmax=1.0,
                   aspect='auto', origin='upper')

    ax.set_xticks(range(len(tex_ids)))
    ax.set_xticklabels(tex_names, rotation=35, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(P_VALS)))
    if ax == axes[0]:
        ax.set_yticklabels(P_LABELS, fontsize=9)
    ax.set_title(LABELS[norm], fontsize=11, fontweight='bold', pad=8)

    for i in range(len(P_VALS)):
        for j in range(len(tex_ids)):
            v = mat[i, j]
            if not np.isnan(v):
                color = 'white' if v < 0.3 or v > 0.75 else '#222'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

plt.colorbar(im, ax=axes[-1], label='Recall (test)', fraction=0.04, pad=0.04)
plt.tight_layout()
plt.savefig(OUT / 'heatmap_lp_norm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Heatmap sauvée")

# ─── Figure 2 : courbes recall vs p, une par texture ──────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
fig.suptitle('Recall (test) vs ordre p de la norme LP — par texture',
             fontsize=12, y=1.01)

x = np.arange(len(P_VALS))

for ax, tid in zip(axes.flat, tex_ids):
    for norm in NORMS:
        vals = [data[norm].get(p, {}).get(tid, {}).get('test', np.nan)
                for p in P_VALS]
        ax.plot(x, vals, color=COLORS[norm], lw=2.0, marker='o',
                markersize=4, label=LABELS[norm])

    ax.axhline(0.5, color='#ccc', lw=0.8, ls=':')
    ax.set_title(TEXTURES[tid], fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(P_LABELS, fontsize=8, rotation=30, ha='right')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis='y', alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

# Dernier subplot vide → légende
axes.flat[-1].set_visible(False)
handles = [plt.Line2D([0], [0], color=COLORS[n], lw=2, label=LABELS[n])
           for n in NORMS]
fig.legend(handles=handles, loc='lower right', fontsize=10,
           bbox_to_anchor=(0.98, 0.08))

for ax in axes[:, 0]:
    ax.set_ylabel('Recall (test)', fontsize=9)

plt.tight_layout()
plt.savefig(OUT / 'curves_lp_norm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Courbes sauvées")

# ─── Figure 3 : meilleur recall par texture et par normalisation ───────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle('Meilleur recall test (sur tous les p) par texture et normalisation', fontsize=12)

n_tex = len(tex_ids)
n_nrm = len(NORMS)
w = 0.20
offsets = np.linspace(-(n_nrm - 1) * w / 2, (n_nrm - 1) * w / 2, n_nrm)

for ci, norm in enumerate(NORMS):
    bests = []
    best_ps = []
    for tid in tex_ids:
        vals = [(data[norm].get(p, {}).get(tid, {}).get('test', 0) or 0, p)
                for p in P_VALS]
        best_v, best_p = max(vals, key=lambda x: x[0])
        bests.append(best_v)
        best_ps.append(best_p)

    bars = ax.bar(np.arange(n_tex) + offsets[ci], bests, w - 0.02,
                  color=COLORS[norm], label=LABELS[norm],
                  edgecolor='white', linewidth=0.5)

    for bar, bp, bv in zip(bars, best_ps, bests):
        if bv > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bv + 0.015, f'p={bp}',
                    ha='center', va='bottom', fontsize=6.5, color='#333')

ax.set_xticks(np.arange(n_tex))
ax.set_xticklabels(tex_names, fontsize=10)
ax.set_ylabel('Recall test (max sur tous les p)', fontsize=10)
ax.set_ylim(0, 1.18)
ax.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig(OUT / 'best_per_texture_lp_norm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Barres groupées sauvées")
