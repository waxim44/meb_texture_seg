#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path('/home/aidouni/meb_texture_seg/output_ouassim/vlad_vs_mean')

TEXTURES = ['Tot.homogène', 'Faisceaux', 'Filaments', 'Strat.rect',
            'Strat.sin', 'Granuleux', 'Trou']
GAINS    = [-0.117, +0.072, -0.111, +0.033, -0.053, -0.027, -0.035]
SIL      = 0.029

x = np.arange(len(TEXTURES))
colors = ['#4C72B0' if g >= 0 else '#C44E52' for g in GAINS]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(x, GAINS, color=colors, alpha=0.85, width=0.6)
ax.axhline(0, color='black', lw=0.8)
ax.axhline(0.05, color='gray', lw=0.8, ls='--')

ax.set_xticks(x)
ax.set_xticklabels(TEXTURES, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Gain recall VLAD − Mean', fontsize=10)
ax.set_title(f'VLAD vs Mean — b7 — K=8 — LOIO recall\nSilhouette K-means={SIL:.3f}',
             fontsize=10)
ax.spines[['top', 'right']].set_visible(False)

for bar, g in zip(bars, GAINS):
    ypos = g + 0.005 if g >= 0 else g - 0.012
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'{g:+.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
out = OUT / 'gain_vlad_vs_mean_clean.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Sauvé : {out}")
