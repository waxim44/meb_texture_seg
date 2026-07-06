#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TNAMES = {1:'Tot.homogène', 3:'Faisceaux', 4:'Filaments', 5:'Strat.rect',
          6:'Strat.sin', 7:'Granuleux', 9:'Trou'}

best = {
    1: {'bloc':'block_12',    'mean':1.000, 'std':0.000},
    7: {'bloc':'block_7',     'mean':0.860, 'std':0.219},
    4: {'bloc':'block_9',     'mean':0.837, 'std':0.323},
    3: {'bloc':'stage_3_fpn', 'mean':0.790, 'std':0.278},
    9: {'bloc':'stage_1_fpn', 'mean':0.735, 'std':0.388},
    6: {'bloc':'block_9',     'mean':0.524, 'std':0.383},
    5: {'bloc':'block_0',     'mean':0.502, 'std':0.498},
}

order      = sorted(best, key=lambda t: -best[t]['mean'])
tex_labels = [TNAMES[t] for t in order]
means      = np.array([best[t]['mean'] for t in order])
stds       = np.array([best[t]['std']  for t in order])
blocs      = [best[t]['bloc'] for t in order]
colors     = ['#2ecc71' if m >= 0.80 else '#f39c12' if m >= 0.60 else '#e74c3c'
              for m in means]

# Barres d'erreur clippées à [0, 1]
err_lo = np.minimum(stds, means)
err_hi = np.minimum(stds, 1.0 - means)

x = np.arange(len(order))
fig, ax = plt.subplots(figsize=(11, 5.5))
bars = ax.bar(x, means, yerr=[err_lo, err_hi], capsize=5, color=colors,
              edgecolor='white', linewidth=0.8,
              error_kw=dict(lw=1.5, capthick=1.5, ecolor='black'))

for bar, bloc, hi in zip(bars, blocs, err_hi):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + hi + 0.025,
            bloc, ha='center', va='bottom', fontsize=8, color='#333')

ax.set_xticks(x)
ax.set_xticklabels(tex_labels, fontsize=11)
ax.set_ylabel('Recall LOIO (one-vs-rest)', fontsize=11)
ax.set_ylim(0, 1.18)
ax.axhline(0.5, color='gray', lw=0.9, ls='--', alpha=0.6)
ax.set_title('Meilleur bloc SAM par texture — Recall LP LOIO', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
plt.savefig('output_ouassim/meilleur_bloc_par_texture.png', dpi=150, bbox_inches='tight')
plt.close()
print("Sauvé : output_ouassim/meilleur_bloc_par_texture.png")
