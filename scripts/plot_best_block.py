#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path('/home/aidouni/meb_texture_seg/output_ouassim')

textures  = ['Tot.homogène', 'Granuleux', 'Filaments', 'Faisceaux',
             'Trou', 'Strat.sin', 'Strat.rect']
recalls   = [0.980, 0.860, 0.837, 0.790, 0.735, 0.524, 0.502]
stds      = [0.001, 0.219, 0.323, 0.278, 0.388, 0.383, 0.498]
blocs     = ['block_12', 'block_7', 'block_9', 'stage_3_fpn',
             'stage_1_fpn', 'block_9', 'block_0']

x = np.arange(len(textures))
colors = ['#2ecc71' if r >= 0.8 else '#f39c12' if r >= 0.6 else '#e74c3c'
          for r in recalls]

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(x, recalls, yerr=stds, capsize=5, color=colors,
              edgecolor='white', linewidth=0.8, error_kw=dict(lw=1.5, capthick=1.5))

for i, (bar, bloc) in enumerate(zip(bars, blocs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.02,
            bloc, ha='center', va='bottom', fontsize=7.5, color='#444444')

ax.set_xticks(x)
ax.set_xticklabels(textures, fontsize=10)
ax.set_ylabel('Recall LOIO (one-vs-rest)', fontsize=11)
ax.set_ylim(0, 1.18)
ax.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_title('Meilleur bloc SAM par texture — Recall LP LOIO', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
fname = OUT / 'best_block_par_texture.png'
plt.savefig(fname, dpi=150, bbox_inches='tight')
print(f"Sauvé : {fname}")
